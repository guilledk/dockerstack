#!/usr/bin/env python3

import copy
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
from docker.types import Mount, IPAMConfig, IPAMPool
from docker.models.images import Image
from docker.models.containers import Container
from docker.models.networks import Network

from dockerstack.utils import (
     docker_pull_image,
    docker_get_running_container,
    docker_stop, download_www_file
)


from .cache import CacheDir
from .errors import DockerStackException
from .typing import ServiceConfig, StackConfig, WWWFileParams
from .service import DockerService
from .logging import DockerStackLogger, get_stack_logger


LOGROTATOR_IMAGE = 'blacklabelops/logrotate:latest'


class DockerStack:

    def __init__(
        self,
        logger: DockerStackLogger | None = None,
        log_level: str = 'INFO',
        root_pwd: Path | None = None,
        config: dict | None = None,
        config_name: str  = 'stack.json',
        cache_dir: str | CacheDir | None = None
    ):
        self.name: str
        self.pid = os.getpid()
        self.client: DockerClient = docker.from_env()
        self.config: StackConfig
        self.network: Network | None

        self.services = SimpleNamespace()
        self.service_configs: dict[str, ServiceConfig] = {}
        self.ordered_services: list[DockerService] = []

        # map service alias -> service name
        self._alias_map: dict[str, str] = {}

        self.logger: DockerStackLogger
        if logger is None:
            self.logger = get_stack_logger(log_level)
        else:
            self.logger = logger

        if not root_pwd:
            self.root_pwd = Path()
        else:
            self.root_pwd = root_pwd

        self.root_pwd.mkdir(parents=True, exist_ok=True)
        self.root_pwd = self.root_pwd.resolve()

        self.cache_dir: CacheDir
        if isinstance(cache_dir, str):
            self.cache_dir = CacheDir(cache_dir)

        elif isinstance(cache_dir, CacheDir):
            self.cache_dir = cache_dir

        elif cache_dir is None:
            self.cache_dir: CacheDir = CacheDir(self.root_pwd / '.cache')

        else:
            raise DockerStackException(f'Invalid cache dir config {cache_dir}')

        self.services_wd = self.root_pwd / 'services'
        self.services_wd.mkdir(exist_ok=True)

        self.logs_wd = self.root_pwd / 'logs'
        self.logs_wd.mkdir(exist_ok=True)

        self.shared_wd = self.root_pwd / 'shared'
        self.shared_wd.mkdir(exist_ok=True)

        self.source_wd = Path(__file__).parent.resolve()

        self.load_config(name=config_name, config=config)

        shared_entrypoint = self.shared_wd / 'entrypoint_wrapper.sh'
        if not shared_entrypoint.is_file():
            shutil.copy(
                self.source_wd / 'entrypoint_wrapper.sh',
                shared_entrypoint
            )

        self.logrotator_name = f'{self.name}-logrotate'
        self.logrotator_image: Image | None = None

        self.logrotator: Container | None = None

    def _generate_alias_maps(self) -> None:
        self._alias_map = {}
        for service in self.config.stack:
            self._alias_map[service['name']] = service['name']
            aliases = service['aliases'] if 'aliases' in service else []
            self._alias_map.update(
                {a: service['name'] for a in aliases})

    def service_alias_to_name(self, alias: str) -> str:
        if alias not in self._alias_map:
            raise DockerStackException(f'Unknown service alias {alias}')

        return self._alias_map[alias]

    def write_config(
        self,
        name: str = 'stack.json',
        target: Path | None = None
    ):
        if not target:
            target = self.root_pwd / name

        config_file = target.resolve()

        with open(config_file, 'w+') as config_file:
            json.dump(
                self.config.model_dump(), config_file, indent=4)

    def load_config(
        self,
        name: str = 'stack.json',
        config: dict[str, Any] | None = None,
        target: Path | None = None
    ):
        if not target:
            target = self.root_pwd / name

        if config is None:
            if not isinstance(target, Path):
                raise DockerStackException(f'Couldn\'t figure out config source')

            with open(target, 'r') as config_file:
                config = json.loads(config_file.read())

        elif isinstance(config, dict):
            config = deepcopy(config)

        # sanity checks
        if not isinstance(config, dict):
            raise DockerStackException('Expected config to be a dict')

        if 'stack' not in config:
            raise DockerStackException('stack field not present on config')

        self.config = StackConfig(**config)
        self.name = self.config.name

        # fill service confs with remote bases as early as posible
        service_configs: list[dict[str, Any]] = deepcopy(config['stack'])
        for service in service_configs:
            if 'base' in service:
                file_name = service['name'] + '.json'
                cache_path = 'library/' + file_name

                base_conf: dict
                if self.cache_dir.file_exists(cache_path):
                        base_conf = self.cache_dir.retrieve_json(cache_path)

                else:
                    # use remote json as service config base
                    base_conf_resp = requests.get(service['base'])
                    base_conf_resp.raise_for_status()

                    base_conf = base_conf_resp.json()

                    self.cache_dir.store_json(base_conf, cache_path)

                service.update(**base_conf)

        # update stack config with filled service confs
        self.config.stack = service_configs

        # darwin arch doesn't support host networking mode...
        if self.config.network or sys.platform == 'darwin':
            self.network = self.network_setup(
                self.config.network
                if self.config.network
                else self.config.name
            )
        else:
            self.network = None

        self._generate_alias_maps()

        self.initialize()

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

        www_files = {}
        if 'www_files' in service_conf:
            # ensure www files are in dir
            for www_file in service_conf['www_files']:
                www_file = WWWFileParams(**www_file)

                target_path = service_wd
                if www_file.target_dir:
                    target_path = Path(www_file.target_dir)

                try:
                    final_path = download_www_file(
                        www_file.url,
                        target_path,
                        rename=www_file.rename,
                        logger=self.logger,
                        cache_dir=self.cache_dir
                    )

                    self.logger.stack_info(f'www file ready at {final_path}')

                    www_files[final_path.name] = final_path

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
            self, service_conf, self.root_pwd, www_files=www_files)


    def get_service(self, alias: str) -> DockerService:
        service_name = self.service_alias_to_name(alias)
        return getattr(self.services, service_name)

    def launch_service_container(
        self,
        service: DockerService
    ):
        service.launch()
        service.pre_start()

        # send SIGUSR2 to continue normal container startup
        service.container.kill(signal=signal.SIGUSR2)

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

    def start(self, exist_ok: bool = False):
        self.logger.stack_info(f'{self.config.name} starting...')

        try:
            for service in self.ordered_services:
                service.load_templates()
                service.configure()
                service.get_image()

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
                    raise DockerStackException(
                        f'Service {service} has status {status} and repair == False')

            self.logger.stack_info(f'{self.config.name} finished start')

        except BaseException:
            self.stop()
            raise

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
        teardown: bool = True
    ):
        self.logger.stack_info('')
        self.start(
            exist_ok=exist_ok
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
