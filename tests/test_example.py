#!/usr/bin/env python3

import pytest

from dockerstack.service import DockerService


@pytest.mark.stack_config(
    from_template='examples/multi_stack',
    exist_ok=False, teardown=True
)
def test_multi_fresh(stack):
    assert stack.status == 'healthy'

    tester = stack.get_service('tester')

    dirs = ['dockerstack-master', 'dockerstack-master-2']
    files = ['mainnet-early-snapshot.bin', 'mainnet-deploy-snapshot.bin']
    sizes = [124010144, 743530477]

    for node in dirs + files:
        assert node in tester.www_files

        path = tester.www_files[node]

        assert path.exists()

        if node in dirs:
            assert path.is_dir()

        if node in files:
            assert path.is_file()
            assert path.stat().st_size == sizes[files.index(node)]


@pytest.mark.stack_config(
    from_template='examples/multi_stack',
    target_dir='tests/multi_stack',
    exist_ok=False, teardown=False
)
def test_multi_keep_alive_start(fresh_target_dir, stack):
    assert stack.status == 'healthy'

@pytest.mark.stack_config(
    from_dir='tests/multi_stack',
    exist_ok=True, teardown=False
)
def test_multi_keep_alive_make_redis_unhealty(stack):
    redis: DockerService = stack.get_service('redis')
    redis.stop()

    assert not redis.running
    assert redis.status == 'unhealthy'
    assert stack.status == 'unhealthy'

@pytest.mark.stack_config(
    from_dir='tests/multi_stack',
    exist_ok=True, teardown=False
)
def test_multi_keep_alive_fix_unhealty(stack):
    assert stack.status == 'healthy'


@pytest.mark.stack_config(
    from_dir='tests/multi_stack',
    exist_ok=True, teardown=True
)
def test_multi_keep_alive_stop(stack):
    assert stack.status == 'healthy'

