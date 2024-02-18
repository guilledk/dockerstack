#!/usr/bin/env python3

import copy
import shutil

from typing import Generator
from pathlib import Path

import pdbp
import pytest

from dockerstack.stack import DockerStack


def maybe_get_marker(request, mark_name: str, field: str, default):
    mark = request.node.get_closest_marker(mark_name)
    if mark is None:
        return default
    else:
        return getattr(mark, field)


DEFAULT_CONF = {
    'target_dir': None,
    'exist_ok': False,
    'repair': True,
    'teardown': True,
    'from_example': None,
    'from_dir': None
}


@pytest.fixture
def stack(tmp_path_factory, request) -> Generator[DockerStack, None, None]:

    tmp_path: Path = tmp_path_factory.getbasetemp() / request.node.name

    _stack_config = maybe_get_marker(
        request, 'stack_config', 'kwargs', {})

    stack_config = copy.copy(DEFAULT_CONF)
    stack_config.update(_stack_config)

    # target_dir: root_pwd of the target stack
    target_dir = stack_config['target_dir']

    if target_dir is None:
        target_dir = str(tmp_path)

    target_dir = Path(target_dir)
    assert isinstance(target_dir, Path)

    # exist_ok: dont fail if stack is already up
    exist_ok: bool = stack_config['exist_ok']

    # repair: attempt to restart unhealthy services on attach
    repair: bool = stack_config['repair']

    # teardown: if not True stack will be stopped after tests
    teardown: bool = stack_config['teardown']

    # from_example: creates a fresh stack dir from a template in examples dir
    from_example: str | None = stack_config['from_example']

    # from_dir: create or attach to a stack in a specific dir
    from_dir: str | None = stack_config['from_dir']

    if from_example:
        examples_dir = Path('examples/')
        src_dir: Path = examples_dir / from_example

        if not src_dir.exists() or not src_dir.is_dir():
            pytest.fail(f'Specified example directory does not exist: {src_dir}')

        target_dir /= from_example

        if not target_dir.exists():
            shutil.copytree(src_dir, target_dir)

    if from_dir:
        target_dir = Path(from_dir)

    _stack = DockerStack(
        root_pwd=target_dir.resolve(strict=True))

    with _stack.open(
        exist_ok=exist_ok,
        repair=repair,
        teardown=teardown
    ) as _stack:
        yield _stack
