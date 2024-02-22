#!/usr/bin/env python3

from __future__ import annotations

from string import Template
from typing import Any, Callable, Iterator, Generator
from pathlib import Path
from datetime import datetime

import docker.errors as docker_errors
from docker.models.images import Image

from docker.types import LogConfig, Mount
from docker.models.containers import Container
from pydantic import BaseModel

from .utils import (
    docker_build_image, docker_get_running_container, docker_pull_image, flatten, stream_file, write_templated_file,
    docker_open_process, docker_wait_process, docker_stream_logs, docker_stop
)
from .errors import DockerServiceError
from .typing import ServiceConfig, MkdirEntryDict, StartupKwargs
from .logging import DockerStackLogger


DEFAULT_DOCKER_LABEL = {'created-by': 'dockerstack'}
DEFAULT_FILTER = {'label': DEFAULT_DOCKER_LABEL}


# if frase returns True count as ok startup
# else assume error and throw
PhraseHandler = Callable[[], None]

class PhraseHandlerEntry(BaseModel):
    handler: PhraseHandler | None
    description: str


class DockerService:

    def __init__(
        self,
        stack: 'DockerStack',
        config: ServiceConfig,
        root_pwd: Path,
        www_files: dict[str, Path] = {}
    ):
        self.stack: 'DockerStack' = stack
        self.name: str = config.name
        self.tag: str = config.tag if config.tag is not None else 'dockerstack'
        self.config: ServiceConfig = config
        self.root_pwd: Path = root_pwd
        self.logger: DockerStackLogger = stack.logger
        self.ip = '127.0.0.1' if not self.config.virtual_ip else self.config.virtual_ip
        self.www_files = www_files

        self.template_whitelist: list[str | tuple[str, str]] = []
        self.templates: dict[str, Template] = {}

        self.more_params: dict[str, Any] = {}
        self.environment: dict[str, str] = config.env
        self.www_files: dict[str, Path] = www_files

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

        self.shell = config.shell
        self.user = config.user
        self.group = config.group

        serv_path: str = self.name
        if isinstance(config.service_path, str):
            serv_path = config.service_path

        self.services_wd: Path = root_pwd / 'services'
        self.service_wd: Path = self.services_wd / serv_path

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

        self.phrase_handlers: dict[str, PhraseHandlerEntry] = {}

        default_ok_desc = 'phrase indicates correct startup'
        self.startup_phrases: list[tuple[str, str]] = []
        if isinstance(config.startup_phrase, str):
            self.startup_phrases = [
                (config.startup_phrase, default_ok_desc)
            ]

        elif isinstance(config.startup_phrase, list):
            self.startup_phrases = [
                (phrase, default_ok_desc)
                if isinstance(phrase, str)
                else phrase
                for phrase in config.startup_phrase
            ]

        for phrase, description in self.startup_phrases:
            self.phrase_handlers[phrase] = PhraseHandlerEntry(
                handler=None, description=description)

        self.startup_logs_kwargs: StartupKwargs = config.startup_logs_kwargs

    def __str__(self) -> str:
        return self.name

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

    def configure(self) -> None:
        '''generates fresh config files from templates in
        service dir
        '''

        self.config_subst['timestamp'] = str(datetime.now())
        for templ_path, template in self.templates.items():
            target = self.service_wd / templ_path
            self.logger.stack_info(f'generating {target}...')
            write_templated_file(target, template, self.config_subst)
            self.logger.stack_info('done')

    def _build(self, **kwargs) -> Image:
        '''internal, used to build service image, only use
        if tag & config.docker_file is set
        '''

        try:
            return docker_build_image(
                self.stack.client,
                tag=self.container_image,
                path=str(self.service_wd),
                dockerfile=self.config.docker_file,
                log_fn=self.logger.stack_info if self.config.show_build else None,
                **kwargs
            )

        except RuntimeError as e:
            self.logger.stack_error(str(e))
            raise DockerServiceError(
                f'Couldn\'t build service {self}'
            )

    def _pull(self) -> Image:
        '''internal, used to pull service image, only use
        if config.docker_image is set
        '''

        try:
            return docker_pull_image(
                self.stack.client,
                self.container_image,
                log_fn=self.logger.stack_info if self.config.show_build else None,
            )

        except RuntimeError as e:
            self.logger.stack_error(str(e))
            raise DockerServiceError(
                f'Couldn\'t pull service {self}'
            )

    def get_image(self, **kwargs) -> Image:
        '''ensure service image (be locally built or pulled from
        remote is present on local image repo
        '''

        image = None
        self.logger.stack_info(f'obtaining docker image for {self}')
        if self.config.docker_file:
            image = self._build(**kwargs)

        elif self.config.docker_image:
            image = self._pull()

        else:
            raise DockerServiceError(
                f'Couldn\'t figure out how to get docker image for {self}')

        self.logger.stack_info(f'got docker image for {self}, {image}')

        return image


    def prepare(self) -> None:
        '''STAGE 1:
        ensure all required services are running & set dynamic launch opts
        '''
        service: DockerService | None = None
        try:
            for req_alias in self.config.requires:
                service = self.stack.get_service(req_alias)
                assert isinstance(service, DockerService) and service.running

        except AssertionError:
            raise DockerServiceError(
                f'Required service {service} not running!')

    def launch(self) -> None:
        '''STAGE 2:
        launch container image and wait sigusr2 trap is up
        '''

        # check if there already is a container running from that image
        found = self.stack.client.containers.list(
            filters={'name': self.container_name, 'status': 'running'})

        if len(found) > 0:
            raise DockerServiceError(
                f'Container from image \'{self.container_name}\' is already running.')

        # check if image is present
        try:
            cont_image = self.stack.client.images.get(self.container_image)

        except docker_errors.ImageNotFound:
            raise DockerServiceError(f'Image \'{self.container_image}\' not found.')

        kwargs = {**self.more_params}
        if self.stack.network:
            # set to bridge, and connect to our custom virtual net after Launch
            # this way we can set the ip addr
            kwargs['network'] = str(self.stack.network)

        else:
            kwargs['network'] = 'host'

        # always override entrypoint with our custom one
        # which waits for USR2 before launching the original entrypoint
        # or command.
        cmd: list[str] = cont_image.attrs['Config']['Cmd']
        entrypoint: list[str] = cont_image.attrs['Config']['Entrypoint']

        wrap_exec: list[str] = []
        if self.command:
            wrap_exec = self.command

        elif self.config.entrypoint and len(self.config.entrypoint) > 0:
            wrap_exec = self.config.entrypoint

        elif entrypoint and cmd:
            wrap_exec = entrypoint + cmd

        elif entrypoint:
            wrap_exec = entrypoint

        elif cmd:
            wrap_exec = cmd

        if len(wrap_exec) == 0:
            raise DockerServiceError(
                f'Could not find entrypoint or cmd for service {self.name}')

        wrapped_exec = [
            self.config.shell,
            '-c',
            ' '.join(['/shared/entrypoint_wrapper.sh', self.user, *wrap_exec])
        ]

        # run container
        self.logger.stack_info(f'launching {self.container_name}...')
        container = self.stack.client.containers.run(
            self.container_image,
            command=None,
            name=self.container_name,
            entrypoint=wrapped_exec,
            mounts=list(self.mounts.values()),
            environment=self.environment,
            detach=True,
            log_config=LogConfig(
                type=LogConfig.types.JSON,
                config={'max-size': '100m'}),
            labels=DEFAULT_DOCKER_LABEL,
            remove=True,
            user='root',
            **kwargs
        )

        # sanity check
        if not isinstance(container, Container):
            raise DockerServiceError(f'Couldn\'t get container instance after launch')

        self.container = container

        container.reload()
        self.logger.stack_info(f'immediate status: {container.status}')

        # wait for entrypoint to setup trap
        for chunk in docker_stream_logs(container.id):
            if 'Waiting for SIGUSR2 signal...' in chunk.decode():
                break


    def pre_start(self):
        '''STAGE 3:
        with container up and ready and running as root, perform
        permission setting procedure
        '''

        # create service directories
        paths = [location for location in self.mkdirs]
        if len(paths) > 0:
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

    def _maybe_match_msg(self, msg: str) -> str:
        for phrase, entry in self.phrase_handlers.items():
            if phrase in msg:
                if entry.handler is not None:
                    entry.handler()

                return phrase

        return ''

    def wait_startup(self):
        '''STAGE 4:
        stream logs and optionally wait for startup or a configured phrase handler
        '''
        if len(self.phrase_handlers) > 0 and self.config.wait_startup:

            max_wait_time = self.startup_logs_kwargs.timeout

            self.logger.stack_info(
                f'waiting at most {max_wait_time}s to start until '
                f'one of the following phrases is present on {self}\'s logs:'
            )

            for phrase, entry in self.phrase_handlers.items():
                self.logger.stack_info(
                    f'\"{phrase}\": {entry.description}')

            found_phrase: bool = False
            for msg in self.stream_logs(**self.startup_logs_kwargs.model_dump()):
                if self.config.show_startup:
                    self.logger.stack_info(msg.rstrip())

                maybe_phrase = self._maybe_match_msg(msg)
                if len(maybe_phrase) > 0:
                    self.logger.stack_info(f'phrase {maybe_phrase} found in logs')
                    found_phrase = True
                    break

            if not found_phrase:
                raise DockerServiceError(
                    f'timed out waiting for startup phrase matching')

    def start(self):
        '''STAGE 5:
        maybe run some code right after service is up, start wont be called until
        startup phrase is found on logs if config.wait_startup == True
        '''
        ...

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

    def restart(self, recursive: bool = False) -> None:
        self.stop()

        # check that required services are healthy
        for req_alias in self.config.requires:
            req_service = self.stack.get_service(req_alias)

            self.logger.stack_info(
                f'checking required service {req_service} is running & healthy...')

            if not req_service.running or not req_service.status == 'healthy':
                if recursive:
                    self.stack.restart_service(req_service)

                else:
                    raise DockerServiceError(
                        f'Tried to restart {self} but required service {req_service}'
                        ' is unhealthy and recursive == False')

        self.load_templates()
        self.configure()
        self.get_image()

        self.stack.launch_service(self)

    def stream_logs(self, **kwargs) -> Generator[str, None, None]:
        if not isinstance(self.container, Container):
            raise DockerServiceError(
                f'Tried to stream logs but container is {self.container}')

        if self.log_file:
            log_file = self.stack.logs_wd / self.log_file
            for msg in stream_file(log_file, **kwargs):
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

    @property
    def running(self) -> bool:
        if hasattr(self, 'container') and self.container:
            try:
                self.container.reload()
                return self.container.status == 'running'

            except docker_errors.NotFound:
                ...

        return False

    @property
    def status(self) -> str:
        return 'healthy'

    def register_phrase_handler(
        self,
        fn: PhraseHandler | None,
        phrase: str,
        description: str
    ) -> None:
        if phrase in self.phrase_handlers:
            raise DockerServiceError(
                f'Phrase handler for \"{phrase}\" already exists!')

        self.phrase_handlers[phrase] = PhraseHandlerEntry(
            handler=fn, description=description)


    def phrase_handler(
        self,
        phrase: str,
        description: str
    ):
        '''
        A decorator factory that takes a phrase and returns a decorator.
        The decorator adds the given function to the handlers dictionary with the phrase as key.
        '''
        def decorator(fn):
            self.register_phrase_handler(fn, phrase, description)
            return fn

        return decorator
