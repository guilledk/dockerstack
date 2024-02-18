#!/usr/bin/env python3

from typing import Any, Literal

import pydantic

from pydantic import BaseModel, Field, model_validator


class MountDict(BaseModel):
    name: str
    source: str
    target: str
    mtype: str = Field('bind', alias='type')

class MkdirEntryDict(BaseModel):
    owner: str | None = None
    owner_group: str | None = None
    permissions: str = '644'


class CommonDict(BaseModel):
    name: str
    aliases: list[str] | None = None
    tag: str | None = None
    service_path: str
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
    virtual_ip: str | None = None
    startup_phrase: str | None = None
    startup_logs_kwargs: dict[str, Any] = {'lines': 0, 'from_latest': True, 'timeout': 10}
    wait_startup: bool = True
    show_startup: bool = False
    show_build: bool = True
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


class LogsConfig(BaseModel):
    dirs: list[str] = [ '/logs' ]
    interval: str = 'daily'
    copies: int = 14
    max_size: str = ''
    compression: Literal['compress'] | Literal['nocompress'] = 'compress'


class ConfigDict(BaseModel):
    name: str
    services: list[str]
    network: str | None = None
    stack: list[dict[str, Any]]
    logs: LogsConfig = LogsConfig()
