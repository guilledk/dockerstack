#!/usr/bin/env python3

from typing import Any, Literal

import pydantic

from pydantic import BaseModel, Field, model_validator


class MountDict(BaseModel):
    name: str
    source: str
    target: str
    mtype: str = Field('bind', alias='type')


class StartupKwargs(BaseModel):
    lines: int = 100
    from_latest: bool = True
    timeout: int = 30


class MkdirEntryDict(BaseModel):
    owner: str | None = None
    owner_group: str | None = None
    permissions: str = '644'


class WWWFileParams(BaseModel):
    url: str
    target_dir: str | None = None
    rename: str | None = None
    decompress: bool = True


class ServiceConfig(BaseModel):
    name: str
    base: str | None = None
    aliases: list[str] | None = None
    tag: str | None = None
    service_path: str | None = None
    log_file: str | None = None
    docker_image: str | None = None
    docker_file: str | None = None
    entrypoint: list[str] | None = None
    user: str = 'root'
    group: str = 'root'
    shell: str = '/bin/bash'
    mounts: list[MountDict] = []
    ports: dict[str, int] = {}
    env: dict[str, str] = {}
    mkdirs: dict[str, MkdirEntryDict] = {}
    sym_links: list[tuple[str, str]] = []
    www_files: list[WWWFileParams] = []
    virtual_ip: str | None = None

    startup_phrase: str | list[str | tuple[str, str]] | None = None
    startup_logs_kwargs: StartupKwargs = StartupKwargs()

    wait_startup: bool = True
    show_startup: bool = False
    show_build: bool = False
    requires: list[str] = []
    stop_sequence: list[str] = ['SIGINT', 'SIGTERM', 'SIGKILL']

    model_config: pydantic.ConfigDict = {
        'extra': 'allow'
    }

    @model_validator(mode='before')
    @classmethod
    def ensure_valid_docker_source(cls, values):
        if (('docker_image' not in values or not values['docker_image']) and
            ('docker_file' not in values or not values['docker_file'])):
            raise ValueError(
                f'Couldn\'t figure out docker source for service {values.name}')

        return values


class LogrotateConfig(BaseModel):
    enabled: bool = False
    dirs: list[str] = [ '/logs' ]
    interval: str = 'daily'
    copies: int = 14
    max_size: str = ''
    compression: Literal['compress'] | Literal['nocompress'] = 'compress'


class StackConfig(BaseModel):
    name: str
    services: list[str]
    network: str | None = None
    logs: LogrotateConfig = LogrotateConfig()
    cache_dir: str = '~/.cache/dockerstack'
    stack: list[dict[str, Any]]
