#!/usr/bin/env python3

from __future__ import annotations

import subprocess

from abc import ABC, abstractproperty
from string import Template
from typing import Any, Iterator, Generator
from pathlib import Path
from datetime import datetime

import docker.errors as docker_errors

from docker.types import Mount
from docker.models.containers import Container

from .utils import (
    docker_get_running_container, flatten, write_templated_file,
    docker_open_process, docker_wait_process, docker_stream_logs, docker_stop
)
from .errors import DockerServiceError
from .typing import CommonDict, MkdirEntryDict
from .logging import DockerStackLogger


class DockerService(ABC):

    def __init__(
        self,
        stack: 'DockerStack',
        config: CommonDict,
        root_pwd: Path
    ):
        self.stack: 'DockerStack' = stack
        self.name: str = config.name
        self.tag: str = config.tag if config.tag is not None else 'dockerstack'
        self.config: CommonDict = config
        self.root_pwd: Path = root_pwd
        self.logger: DockerStackLogger = stack.logger

        self.template_whitelist: list[str | tuple[str, str]] = []
        self.templates: dict[str, Template] = {}

        self.more_params: dict[str, Any] = {}
        self.environment: dict[str, str] = config.env

        self.container: Container | None = None
        self.container_name: str = f'{self.stack.name}-{config.name}'
        self.container_image: str = f'{self.tag}-{self.stack.name}'
        if config.docker_image:
            self.container_image = config.docker_image

        self.container = docker_get_running_container(
            self.stack.client, self.container_name)

        if (isinstance(self.container, Container) and
            self.container_image not in self.container.image.attrs['RepoTags']):
            raise DockerServiceError(
                f'Found container with name {self.container_name} but '
                f'its image ({self.container.image}) is not {self.container_image}')

        self.command: list[str] | None = None

        self.startup_logs_kwargs: dict[str, Any] = config.startup_logs_kwargs

        self.shell = config.shell
        self.user = config.user
        self.group = config.group

        self.services_wd: Path = root_pwd / 'services'
        self.service_wd: Path = self.services_wd / config.service_path

        self.log_file: str = f'{self.name}.log'
        if isinstance(config.log_file, str):
            self.log_file = config.log_file

        flat_config = flatten(self.name, config.model_dump())
        port_prefix = f'{self.name}_ports'
        flat_config.update(flatten(port_prefix, flat_config[port_prefix]))

        self.config_subst: dict[str, Any] = flat_config

        # setup mounts
        self.mounts: dict[str, Mount] = {}

        for mount in config.mounts:
            source = mount.source
            if not Path(source).is_absolute():
                source =  str(self.stack.root_pwd.resolve() / source)

            source = Path(source)
            source.mkdir(exist_ok=True)

            self.mounts[mount.name] = Mount(
                mount.target, str(source), mount.mtype)

        if '~' not in self.mounts:
            home_dir = '/root'
            if self.user != 'root':
                home_dir = f'/home/{self.user}'

            self.mounts['~'] = Mount(home_dir, str(self.service_wd.resolve()), 'bind')

        if 'logs' not in self.mounts and self.log_file:
            self.mounts['logs'] = Mount('/logs', str(self.stack.logs_wd.resolve()), 'bind')

        self.mounts['shared'] = Mount('/shared', str(self.stack.shared_wd.resolve()), 'bind')

        # setup default env vars
        if self.log_file:
            self.environment['DS_LOG_FILE'] = self.log_file

        self.environment['SHELL'] = config.shell

        self.ports: dict[str, int] = config.ports

        if self.stack.network and len(config.ports) > 0:
            port_conf = {}
            for pnum in config.ports.values():
                port_conf[str(pnum)] = pnum

            self.more_params['ports'] = port_conf

        self.mkdirs: dict[str, MkdirEntryDict] = config.mkdirs

        for mount in config.mounts:
            if mount.target not in self.mkdirs:
                self.mkdirs[mount.target] = MkdirEntryDict(
                    owner=self.user,
                    owner_group=self.group
                )

        self.sym_links: list[tuple[str, str]] = config.sym_links

    def __str__(self) -> str:
        return self.name

    @property
    def startup_phrase(self) -> str | None:
        return self.config.startup_phrase

    def load_templates(self):
        for template in self.template_whitelist:
            tsrc, tdst = ('', '')
            if isinstance(template, tuple):
                tsrc, tdst = template
            else:
                tsrc, tdst = (template + '.template', template)

            with open(self.service_wd / tsrc, 'r') as templ_file:
                self.templates[tdst] = Template(templ_file.read())

        self.logger.stack_info(f'loaded {len(self.template_whitelist)} {self.name} templates')

    def configure(self):
        self.ip = '127.0.0.1' if not self.config.virtual_ip else self.config.virtual_ip
        self.config_subst['timestamp'] = str(datetime.now())
        for templ_path, template in self.templates.items():
            target = self.service_wd / templ_path
            self.logger.stack_info(f'generating {target}...')
            write_templated_file(target, template, self.config_subst)
            self.logger.stack_info('done')

    def prepare(self):
        service: DockerService | None = None
        try:
            for req_alias in self.config.requires:
                service = self.stack.get_service(req_alias)
                assert isinstance(service, DockerService) and service.running

        except AssertionError:
            raise DockerServiceError(
                f'Required service {service} not running!')

    def pre_start(self):
        # create service directories
        paths = [location for location in self.mkdirs]
        ec, out = self.run_process(['mkdir', '-p', *paths])
        if ec != 0:
            raise DockerServiceError(
                f'Couldn\'t create directories \"{paths}\", mkdir output: {out}')
        self.logger.stack_info(f'created {len(paths)} dirs')

        # set permissions & chown, with exclusions
        exclude_dirs = ['__pycache__']
        exclude_files_patterns = ['runtime.py']
        for path in paths:
            owner = self.mkdirs[path].owner or self.user
            owner_group = self.mkdirs[path].owner_group or self.group
            perms = self.mkdirs[path].permissions

            exclude_dir_cmds = [' '.join(['-name', d, '-prune', '-o']) for d in exclude_dirs]
            exclude_file_cmds = [' '.join(['-name', fp, '-prune', '-o']) for fp in exclude_files_patterns]

            find_cmd = [
                'find', path, '-type d',
                *exclude_dir_cmds, '-true', '-o', '-type f',
                *exclude_file_cmds, '-true',
                '-exec', f'chown {owner}:{owner_group} {{}} +',
                '-exec', f'chmod {perms} {{}} +'
            ]
            ec, out = self.run_in_shell(find_cmd)
            if ec != 0:
                raise DockerServiceError(
                    f'Couldn\'t set perms on \"{path}\", output: {out}')

        self.logger.stack_info(f'set permissions for {len(paths)} dirs')

        # create symlinks
        for link_src, link_dst in self.sym_links:
            ec, out = self.run_in_shell([
                # first guarantee dir exists
                'mkdir', '-p', str(Path(link_dst).parent),
                '&&',
                # make link
                'ln', '-sf', link_src, link_dst
            ])
            if ec != 0:
                raise DockerServiceError(
                    f'Couldn\'t create sym link {(link_src, link_dst)}, {out}')

        self.logger.stack_info(f'created {len(self.sym_links)} symlinks')

        # setup log_file perms if needed
        if self.log_file:
            log_file_guest = f'/logs/{self.log_file}'
            ec, out = self.run_in_shell([
                'touch', log_file_guest,
                '&&',
                'chown', f'{self.user}:{self.group}', log_file_guest
            ])
            if ec != 0:
                raise DockerServiceError(
                    f'Couldn\'t initialize logging file {log_file_guest}')

    def start(self):
        ...

    def _stream_logs_from_dir(
        self,
        timeout: float = 10,
        lines: int = 0,
        from_latest: bool = True
    ) -> Generator[str, None, None]:
        log_path = self.stack.logs_wd / self.log_file
        log_path = log_path.resolve()

        if not log_path.is_file():
            raise DockerServiceError(f'Log file at {log_path} not found!')

        line_str = str(lines) if from_latest else '+1'

        process = subprocess.Popen(
            ['bash', '-c',
                f'timeout {timeout} tail -n {line_str} -f {log_path}'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        for line in iter(process.stdout.readline, b''):
            msg = line.decode()
            yield msg

        process.stdout.close()
        process.wait()

        if process.returncode != 0:
            raise ValueError(
                f'tail returned {process.returncode}\n'
                f'{process.stderr.read().decode("utf-8")}')

    def stream_logs(self, **kwargs) -> Generator[str, None, None]:
        if not isinstance(self.container, Container):
            raise DockerServiceError(
                f'Tried to stream logs but container is {self.container}')

        if self.log_file:
            for msg in self._stream_logs_from_dir(**kwargs):
                yield msg
        else:
            for chunk in docker_stream_logs(self.container.id, **kwargs):
                yield chunk.decode()

    def open_process(self, *args, **kwargs) -> tuple[str, Iterator[bytes]]:
        if not isinstance(self.container, Container):
            raise DockerServiceError(
                f'Tried to open process but container is {self.container}')

        return docker_open_process(
            self.stack.client,
            self.container,
            *args, **kwargs
        )

    def run_process(self, *args, **kwargs) -> tuple[int, str]:
        return docker_wait_process(
            self.stack.client,
            *(self.open_process(*args, **kwargs)))

    def run_in_shell(self, cmd, *args, **kwargs) -> tuple[int, str]:
        cmd = [self.shell, '-c', ' '.join(cmd)]
        return self.run_process(cmd, *args, **kwargs)

    def stop(self, **kwargs):
        if not hasattr(self, 'container') or not self.container:
            return

        try:
            docker_stop(
                self.container,
                stop_sequence=self.config.stop_sequence,
                **kwargs
            )
            # self.container.remove()

        except docker_errors.NotFound:
            ...

    @property
    def running(self) -> bool:
        if hasattr(self, 'container') and self.container:
            try:
                self.container.reload()
                return self.container.status == 'running'

            except docker_errors.NotFound:
                ...

        return False

    @abstractproperty
    def status(self) -> str:
        ...
