#!/usr/bin/env python3

import os
import shutil

from typing import Generator
from pathlib import Path

import pdbp
import pytest
import docker

from docker.types import Mount
from pydantic import BaseModel, model_validator
from dockerstack.stack import DockerStack


class StackFixtureError(BaseException):
    ...


def maybe_get_marker(request, mark_name: str, field: str, default):
    mark = request.node.get_closest_marker(mark_name)
    if mark is None:
        return default
    else:
        return getattr(mark, field)


class StackFixtureParamsBase(BaseModel):
    config_name: str = 'stack.json'   # name/alias of stack.json file
    target_dir: str | None = None     # root_pwd of the target stack
    exist_ok: bool = False            # dont fail if stack is already up
    repair: bool = True               # attempt to restart unhealthy services on attach
    teardown: bool = True             # if not True stack will be stopped after tests

    from_template: str | None = None  # creates a fresh stack dir from a template in from_dir
    from_dir: str | None = None       # create or attach to a stack in a specific dir


class StackFixtureParams(StackFixtureParamsBase):

    @model_validator(mode='before')
    @classmethod
    def validate_source(cls, values: dict):
        from_template = values.get('from_template', None)
        from_dir = values.get('from_dir', None)

        if from_template and from_dir:
            raise StackFixtureError(
                'from_template and from_dir are mutually exclusive.')

        elif from_template is None and from_dir is None:
            raise StackFixtureError(
                'either from_template or from_dir must be present on stack config')

        return values

    @staticmethod
    def from_pytest_marks(request) -> 'StackFixtureParams':
        mark_config = maybe_get_marker(
            request, 'stack_config', 'kwargs', {})

        stack_config = StackFixtureParamsBase().model_dump()
        stack_config.update(mark_config)

        return StackFixtureParams(**stack_config)


@pytest.fixture
def fresh_target_dir(request) -> None:
    config = StackFixtureParams.from_pytest_marks(request)

    if not isinstance(config.from_template, str):
        raise StackFixtureError('fresh_target_dir fixture requires from_template config')

    if not isinstance(config.target_dir, str):
        raise StackFixtureError('fresh_target_dir fixture requires target_dir config')

    template_path = Path(config.from_template)
    target_path = Path(config.target_dir).resolve()

    if target_path.exists():
        # * flashes TCD testing bureau badge *: im commandeering this directory
        client = docker.from_env()
        user: int = os.getuid()
        group: int = os.getgid()

        cmd = ['chown', '-R', f'{user}:{group}', '/target']

        client.containers.run(
            'bash', ['bash', '-c', ' '.join(cmd)],
            mounts=[Mount('/target', str(target_path), 'bind')],
            remove=True
        )

        # make sure chown worked
        stat: os.stat_result = target_path.stat()
        assert stat.st_uid == user
        assert stat.st_gid == group

        # delete
        shutil.rmtree(target_path)
        assert not target_path.exists()

    shutil.copytree(template_path, target_path)


@pytest.fixture
def stack(tmp_path_factory, request) -> Generator[DockerStack, None, None]:

    tmp_path: Path = tmp_path_factory.getbasetemp() / request.node.name

    config = StackFixtureParams.from_pytest_marks(request)

    target_dir = config.target_dir

    if target_dir is None:
        target_dir = str(tmp_path) if not config.from_dir else config.from_dir

    target_dir = Path(target_dir)
    assert isinstance(target_dir, Path)

    if config.from_template:
        src_dir = Path(str(config.from_template))

        if not src_dir.exists() or not src_dir.is_dir():
            pytest.fail(f'Specified example directory does not exist: {src_dir}')

        if not target_dir.exists():
            shutil.copytree(src_dir, target_dir)

    _stack = DockerStack(
        root_pwd=target_dir.resolve(strict=True),
        config_name=config.config_name,
        cache_dir='tests/.cache'
    )

    with _stack.open(
        exist_ok=config.exist_ok,
        teardown=config.teardown
    ) as _stack:
        yield _stack
