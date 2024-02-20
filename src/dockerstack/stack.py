#!/usr/bin/env python3

import os
import sys
import json
import shutil
import signal
import inspect
import importlib.util

from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from pathlib import Path
from contextlib import contextmanager

import docker
import docker.errors as docker_errors

import requests
import networkx as nx

from docker import DockerClient
from docker.types import Mount, LogConfig, IPAMConfig, IPAMPool
from docker.models.images import Image
from docker.models.containers import Container
from docker.models.networks import Network

from dockerstack.utils import (
    docker_build_image, docker_pull_image,
    docker_get_running_container,
    docker_stream_logs,
    docker_stop, download_www_file
)


from .errors import DockerStackException
from .typing import ServiceConfig, StackConfig, WWWFileParams
from .service import DockerService
from .logging import DockerStackLogger, get_stack_logger


DEFAULT_DOCKER_LABEL = {'created-by': 'dockerstack'}
DEFAULT_FILTER = {'label': DEFAULT_DOCKER_LABEL}

LOGROTATOR_IMAGE = 'blacklabelops/logrotate:latest'


class DockerStack:

    def __init__(
        self,
        logger: DockerStackLogger | None = None,
        log_level: str = 'INFO',
        root_pwd: Path | None = None,
        config: dict | None = None,
        config_name: str  = 'stack.json'
    ):
        self.pid = os.getpid()
        self.client: DockerClient = docker.from_env()
        self.config: StackConfig
        self.network: Network | None

        self.services = SimpleNamespace()
        self.service_configs: dict[str, ServiceConfig] = {}
        self.ordered_services: list[DockerService] = []

        self.logger: DockerStackLogger
        if logger is None:
            self.logger = get_stack_logger(log_level)
        else:
            self.logger = logger

        if not root_pwd:
            self.root_pwd = Path()
        else:
            self.root_pwd = root_pwd

        self.load_config(name=config_name, config=config)
        self.name: str = self.config.name

        self.root_pwd.mkdir(parents=True, exist_ok=True)
        self.root_pwd = self.root_pwd.resolve()

        self.services_wd = self.root_pwd / 'services'
        self.services_wd.mkdir(exist_ok=True)

        self.logs_wd = self.root_pwd / 'logs'
        self.logs_wd.mkdir(exist_ok=True)

        self.shared_wd = self.root_pwd / 'shared'
        self.shared_wd.mkdir(exist_ok=True)

        self.source_wd = Path(__file__).parent.resolve()

        shared_entrypoint = self.shared_wd / 'entrypoint_wrapper.sh'
        if not shared_entrypoint.is_file():
            shutil.copy(
                self.source_wd / 'entrypoint_wrapper.sh',
                shared_entrypoint
            )

        self.logrotator_name = f'{self.name}-logrotate'
        self.logrotator_image: Image | None = None

        self.logrotator: Container | None = None


    def service_alias_to_name(self, alias: str) -> str:
        name_match: str | None = None
        for service in self.config.stack:
            service_name = service['name']
            if (alias == service_name or
                ('aliases' in service and alias in service['aliases'])):
                name_match = service_name
                break

        if not name_match:
            raise DockerStackException(f'Unknown service alias {alias}')

        return name_match

    def write_config(self, name: str = 'stack.json'):
        config_file = (self.root_pwd / name).resolve()

        with open(config_file, 'w+') as config_file:
            config_file.write(json.dumps(self.config.model_dump(), indent=4))

    def load_config(
        self,
        name: str = 'stack.json',
        config: dict[str, Any] | None = None,
        target: Path | None = None
    ):
        if not target:
            target = self.root_pwd / name

        if not config:
            with open(target, 'r') as config_file:
                config = json.loads(config_file.read())

        else:
            config = deepcopy(config)

        self.config = StackConfig(**config)

        # darwin arch doesn't support host networking mode...
        if self.config.network or sys.platform == 'darwin':
            self.network = self.network_setup(
                self.config.network
                if self.config.network
                else self.config.name
            )
        else:
            self.network = None


    def _get_raw_service_config(
        self,
        alias: str
    ) -> dict[str, Any]:
        service_name = self.service_alias_to_name(alias)
        service_conf = None
        for service in self.config.stack:
            if service['name'] == service_name:
                service_conf = service
                break

        if not service_conf:
            raise DockerStackException(f'Raw config for service {service_name} not found!')

        if 'base' in service_conf:
            # define cache directory
            cache_dir = (
                Path(self.config.cache_dir) / 'library').expanduser().resolve()
            cache_dir.mkdir(parents=True, exist_ok=True)

            file_name = service_conf['name'] + '.json'

            cache_file_path = (cache_dir / file_name).resolve()

            base_conf: dict
            if cache_file_path.exists():
                # load from cache
                with open(cache_file_path, 'r') as cache_file:
                    base_conf = json.load(cache_file)

            else:
                # use remote json as service config base
                base_conf_resp = requests.get(service_conf['base'])
                base_conf_resp.raise_for_status()

                base_conf = base_conf_resp.json()

                with open(cache_file_path, 'w+') as cache_file:
                    cache_file.write(json.dumps(base_conf, indent=4))

            service_conf.update(**base_conf)

        return service_conf

    def get_service_config(self, alias: str) -> ServiceConfig:
        service_name = self.service_alias_to_name(alias)

        if service_name not in self.service_configs:
            raise DockerStackException(f'Config for service {service_name} not found!')

        return self.service_configs[service_name]

    def construct_service(self, alias: str) -> DockerService:
        service_name = self.service_alias_to_name(alias)
        service_conf = self._get_raw_service_config(alias)

        serv_path = service_name
        if 'service_path' in service_conf:
            serv_path = service_name

        service_wd = self.services_wd / serv_path
        service_wd.mkdir(parents=True, exist_ok=True)

        if 'www_files' in service_conf:
            # ensure www files are in dir
            for www_file in service_conf['www_files']:
                www_file = WWWFileParams(**www_file)

                target_path = service_wd
                if www_file.target_dir:
                    target_path = Path(www_file.target_dir)

                try:
                    download_www_file(
                        www_file.url,
                        target_path,
                        logger=self.logger,
                        cache_dir_path=Path(self.config.cache_dir) / 'files'
                    )

                except BaseException as e:
                    raise DockerStackException(f'Failed to download {www_file.url}, {e}')

        service_spec_path = service_wd / 'runtime.py'
        service_class = DockerService
        service_conf_class = ServiceConfig

        if service_spec_path.exists():
            spec = importlib.util.spec_from_file_location(service_name, service_spec_path)
            if spec is None:
                raise ImportError(f"Could not load spec for {alias} from {service_wd}")

            module = importlib.util.module_from_spec(spec)
            if spec.loader is not None:
                spec.loader.exec_module(module)

            else:
                raise ImportError(f"Failed to load module {alias} from {service_wd}")

            for _, obj in inspect.getmembers(module):
                if inspect.isclass(obj):
                    if issubclass(obj, DockerService):
                        service_class = obj

                    elif issubclass(obj, ServiceConfig) and obj is not ServiceConfig:
                        service_conf_class = obj

        service_conf = service_conf_class(**service_conf)
        self.service_configs[service_name] = service_conf

        return service_class(
            self, service_conf, self.root_pwd)


    def get_service(self, alias: str) -> DockerService:
        service_name = self.service_alias_to_name(alias)
        return getattr(self.services, service_name)

    def pull_service(self, alias: str) -> Image:
        service: DockerService = self.get_service(alias)

        try:
            return docker_pull_image(
                self.client,
                service.container_image,
                log_fn=self.logger.stack_info
            )

        except RuntimeError as e:
            self.logger.stack_error(str(e))
            raise DockerStackException(
                f"Couldn't build service {alias}."
            )

    def build_service(self, alias: str, **kwargs) -> Image:
        service: DockerService = self.get_service(alias)

        try:
            return docker_build_image(
                self.client,
                tag=service.container_image,
                path=str(service.service_wd),
                dockerfile=service.config.docker_file,
                log_fn=self.logger.stack_info,
                **kwargs
            )

        except RuntimeError as e:
            self.logger.stack_error(str(e))
            raise DockerStackException(
                f"Couldn't build service {alias}."
            )

    def get_service_image(self, alias: str, **kwargs) -> Image:
        service = self.get_service(alias)

        if service.config.docker_file:
            return self.build_service(alias, **kwargs)

        if service.config.docker_image:
            return self.pull_service(alias)

        raise DockerStackException(
            f'Couldn\'t figure out how to get docker image for {service.name}')

    def launch_service_container(
        self,
        service: DockerService
    ):
        # check if there already is a container running from that image
        found = self.client.containers.list(
            filters={'name': service.container_name, 'status': 'running'})

        if len(found) > 0:
            raise DockerStackException(
                f'Container from image \'{service.container_name}\' is already running.')

        # check if image is present
        try:
            cont_image = self.client.images.get(service.container_image)

        except docker_errors.ImageNotFound:
            raise DockerStackException(f'Image \'{service.container_image}\' not found.')

        kwargs = deepcopy(service.more_params)
        if self.network:
            # set to bridge, and connect to our custom virtual net after Launch
            # this way we can set the ip addr
            kwargs['network'] = 'bridge'

        else:
            kwargs['network'] = 'host'

        # always override entrypoint with our custom one
        # which waits for USR2 before launching the original entrypoint
        # or command.
        cmd: list[str] = cont_image.attrs['Config']['Cmd']
        entrypoint: list[str] = cont_image.attrs['Config']['Entrypoint']

        wrap_exec: list[str] = []
        if service.command:
            wrap_exec = service.command

        elif service.config.entrypoint and len(service.config.entrypoint) > 0:
            wrap_exec = service.config.entrypoint

        elif entrypoint and cmd:
            wrap_exec = entrypoint + cmd

        elif entrypoint:
            wrap_exec = entrypoint

        elif cmd:
            wrap_exec = cmd

        if len(wrap_exec) == 0:
            raise DockerStackException(
                f'Could not find entrypoint or cmd for service {service.name}')

        wrapped_exec = [
            service.config.shell,
            '-c',
            ' '.join(['/shared/entrypoint_wrapper.sh', service.user, *wrap_exec])
        ]

        # run container
        self.logger.stack_info(f'launching {service.container_name}...')
        container = self.client.containers.run(
            service.container_image,
            command=None,
            name=service.container_name,
            entrypoint=wrapped_exec,
            mounts=list(service.mounts.values()),
            environment=service.environment,
            detach=True,
            log_config=LogConfig(
                type=LogConfig.types.JSON,
                config={'max-size': '100m'}),
            labels=DEFAULT_DOCKER_LABEL,
            remove=True,
            user='root',
            **kwargs
        )
        assert isinstance(container, Container)
        service.container = container

        container.reload()
        self.logger.stack_info(f'immediate status: {container.status}')

        # wait for entrypoint to setup trap
        for chunk in docker_stream_logs(container.id):
            if 'Waiting for SIGUSR2 signal...' in chunk.decode():
                break

        service.pre_start()

        # send SIGUSR2 to continue normal container startup
        container.kill(signal=signal.SIGUSR2)

    def stream_logs(self, source: str, **kwargs):
        for log in getattr(self.services, source).stream_logs(**kwargs):
            yield log

    def network_setup(self, name: str):
        self.logger.stack_info(f'setting up network {name}...')
        try:
            self.network = self.client.networks.get(name)

        except docker_errors.NotFound:
            ipam_pool = IPAMPool(
                subnet='192.168.123.0/24',
                gateway='192.168.123.254'
            )
            ipam_config = IPAMConfig(
                pool_configs=[ipam_pool]
            )

            self.network = self.client.networks.create(
                name, 'bridge', ipam=ipam_config
            )

        self.logger.stack_info(f'network online')

    def network_ip(self, service: str) -> str:
        service = self.service_alias_to_name(service)
        if hasattr(self.services, service):
            return getattr(self.services, service).ip

        if service in self.config.model_dump():
            config: ServiceConfig = getattr(self.config, service)
            return '127.0.0.1' if not config.virtual_ip else config.virtual_ip

        raise AttributeError('Couldn\'t figure out ip for {service}')

    def network_service_setup(self, service: DockerService):
        self.logger.stack_info(f'connecting {service.name} to network {self.network.name}...')
        self.network.connect(
            service.container,
            ipv4_address=service.config.virtual_ip
        )
        self.logger.stack_info(f'{service.name} connected.')

    def launch_service(self, serv: DockerService, exist_ok: bool = True):
        if serv.running:
            if not exist_ok:
                raise DockerStackException(f'Service {serv.name} already launched!')

            # service = getattr(self.services, name)
            self.logger.stack_info(f'tried to launch {serv.name} but already running.')
            return

        self.logger.stack_info(f'{serv.name} not found, initializing...')

        self.logger.stack_info(f'prepare {serv.name}')
        serv.prepare()
        self.launch_service_container(serv)

        if serv.startup_phrase and serv.config.wait_startup:
            phrase = serv.startup_phrase

            self.logger.stack_info(
                f'waiting until phrase \"{phrase}\" is present '
                f'in {serv.name} logs (max wait time: {serv.startup_logs_kwargs.timeout} sec)')

            found_phrase = False
            for msg in serv.stream_logs(**serv.startup_logs_kwargs.model_dump()):
                if serv.config.show_startup:
                    self.logger.stack_info(msg.rstrip())

                if serv.startup_phrase in msg:
                    found_phrase = True
                    self.logger.stack_info('found phrase!')
                    break

            if not found_phrase:
                raise DockerStackException(
                    f'timed out waiting for phrase \"{phrase}\" to be present '
                    f'in {serv.name}\'s logs.')

        if self.network:
            self.network_service_setup(serv)

        if not serv.running:
            raise DockerStackException(
                f'serv: {serv.name} not running.')

        self.logger.stack_info(f'start {serv.name}')
        serv.start()
        self.logger.stack_info(f'started {serv.name}')

    def initialize(self):
        service_requirements: dict[str, set[str]] = {}

        requested_services = [
            self.service_alias_to_name(alias)
            for alias in self.config.services
        ]

        def add_to_reqs(sname: str):
            serv_conf = self._get_raw_service_config(sname)
            _reqs = set()
            if 'requires' in serv_conf:
                _reqs = set(serv_conf['requires'])
                for _req in _reqs:
                    add_to_reqs(_req)

            service_requirements[sname] = _reqs

        for service_name in requested_services:
            add_to_reqs(service_name)

        G = nx.DiGraph(service_requirements)

        try:
            ordered_services_names = list(nx.topological_sort(G))
            ordered_services_names.reverse()

        except nx.NetworkXUnfeasible:
            raise ValueError("Dependency graph has at least one cycle, which means there is no valid launch order!")

        for service_alias in ordered_services_names:
            service_name = self.service_alias_to_name(service_alias)
            service = self.construct_service(service_name)
            self.ordered_services.append(service)
            setattr(self.services, service_name, service)

    def _maybe_setup_logrotate(self):
        self.logrotator_image = docker_pull_image(
            self.client, LOGROTATOR_IMAGE, log_fn=self.logger.stack_info)

        self.logrotator = docker_get_running_container(
            self.client, self.logrotator_name)

        if self.logrotator is None:
            logsconf = self.config.logs
            logsenv = {
                'LOGS_DIRECTORIES': ' '.join(logsconf.dirs),
                'LOGROTATE_INTERVAL': logsconf.interval,
                'LOGROTATE_COPIES': str(logsconf.copies),
                'LOGROTATE_SIZE': logsconf.max_size,
                'LOGROTATE_COMPRESSION': logsconf.compression
            }
            self.logrotator = self.client.containers.run(
                self.logrotator_image,
                name=self.logrotator_name,
                mounts=[Mount('/logs', str(self.logs_wd.resolve()), 'bind')],
                environment=logsenv,
                detach=True,
                remove=True
            )

            self.logrotator.reload()
            if self.logrotator.status != 'running':
                raise DockerStackException(
                    f'Failed to start logrotate service!')

    def _maybe_stop_logrotate(self):
        if self.logrotator:
            docker_stop(self.logrotator)

    def start(self, exist_ok: bool = False, repair: bool = True):
        self.logger.stack_info(f'{self.config.name} starting...')

        try:
            for service in self.ordered_services:
                service.load_templates()
                service.configure()
                self.get_service_image(service.name)

            if self.config.logs.enabled:
                self._maybe_setup_logrotate()

            for service in self.ordered_services:
                if not service.running:
                    self.launch_service(service)

                elif not exist_ok:
                    raise DockerStackException(
                        f'Service already running and exist_ok == False!')

                status = service.status
                if status == 'unhealthy':
                    if not repair:
                        raise DockerStackException(
                            f'Service {service} has status {status} and repair == False')

                    # attempt service restart
                    self.restart_service(service)

                    # if still unhealthy raise
                    if service.status == 'unhealthy':
                        raise DockerStackException(
                            f'Tried to restart unhealthy service {service}')

            self.logger.stack_info(f'{self.config.name} finished start')

        except BaseException:
            self.stop()
            raise

    def restart_service(self, service: DockerService):
        self.logger.stack_info(f'restarting {service}')
        service.stop()
        service.load_templates()
        service.configure()
        self.get_service_image(service.name)

        # check that required services are healthy
        for req_alias in service.config.requires:
            req_service = self.get_service(req_alias)

            self.logger.stack_info(f'checking required service {req_service} is running & healthy...')

            if not req_service.running or not req_service.status == 'healthy':
                self.restart_service(req_service)

        self.launch_service(service)
        self.logger.stack_info(f'restarted {service}')

    def stop(self):
        self.logger.stack_info(f'{self.config.name} is stopping...')
        for service in vars(self.services).values():
            if service.running:
                service.stop()
                self.logger.stack_info(f'stopped {service.name}')

        self._maybe_stop_logrotate()

        self.logger.stack_info(f'{self.config.name} stopped')

    @contextmanager
    def open(
        self,
        exist_ok: bool = False,
        teardown: bool = True,
        repair: bool = True
    ):
        self.logger.stack_info('')
        self.initialize()
        self.start(
            exist_ok=exist_ok,
            repair=repair
        )
        yield self
        if teardown:
            self.stop()

        else:
            self.logger.stack_info(f'skipping teardown...')

    @property
    def status(self) -> str:
        for service in vars(self.services).values():
            if service.status != 'healthy':
                return 'unhealthy'

        return 'healthy'
