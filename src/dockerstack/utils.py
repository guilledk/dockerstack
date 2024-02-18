#!/usr/bin/env python3

import time
import json
import random
import string
import struct
import socket
import logging
import tarfile

from string import Template
from typing import Callable, Iterator, Generator
from pathlib import Path
from docker.models.images import Image
from urllib3.connection import HTTPConnection
from urllib3.connectionpool import HTTPConnectionPool

import requests
import docker.errors as docker_errors

from docker import DockerClient
from docker.models.containers import Container
from requests.adapters import HTTPAdapter


def random_string(size=256) -> str:
    return ''.join(
        random.choice(string.ascii_lowercase + string.digits)
        for _ in range(size)
    )


def flatten(master_key, _dict) -> dict:
    ndict = {}

    for key, val in _dict.items():
        ndict[f'{master_key}_{key}'] = val

    return ndict


def write_templated_file(target_dir: Path, template: Template, subst: dict) -> None:
    with open(target_dir, 'w+') as target_file:
        target_file.write(template.substitute(**subst))

def get_free_port(tries: int = 10) -> int:
    _min = 10000
    _max = 60000

    for _ in range(tries):
        port_num = random.randint(_min, _max)

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            s.bind(("127.0.0.1", port_num))
            s.close()

        except socket.error as e:
            continue

        else:
            return port_num


# docker helpers

def docker_pull_image(
    client: DockerClient,
    image: str,
    log_fn: Callable | None = None
) -> Image:
    """
    Pulls a docker image from a repository.

    Args:
        client (DockerClient): The docker client instance.
        image (str): The repository tag of the image to pull in 'repo:tag' format.
        log_fn (Callable[[str], None] | None, optional): A callback function for logging progress. Defaults to None.

    Raises:
        ValueError: If the 'image' does not follow 'repo:tag' format.
        RuntimeError: If the image is not found or there is an API error during pull.

    Returns:
        Image: The pulled docker image object.
    """
    # check if image is present locally and avoid slow api.pull
    try:
        return client.images.get(image)

    except docker_errors.ImageNotFound:
        ...

    # validate image format
    try:
        repo, tag = image.split(':')

    except ValueError:
        raise ValueError(f'Expected \"repo:tag\" format for {image}.')

    # pull the image and log updates
    try:
        updates = {}
        last_status = ''
        for update in client.api.pull(repo, tag=tag, stream=True, decode=True):
            _id = update.get('id')
            status = update.get('status')
            if _id and (status != last_status):
                updates[_id] = status
                last_status = status
                if log_fn:
                    log_fn(f'{_id}: {status}')

        img = client.images.get(image)
        assert isinstance(img, Image)
        return img

    except docker_errors.ImageNotFound:
        raise RuntimeError(f'Image \"{image}\" not found.')
    except docker_errors.APIError as api_error:
        raise RuntimeError(f'Failed to pull the image due to an API error. {api_error}')


def docker_build_image(
    client: DockerClient,
    tag: str,
    path: str,
    log_fn: Callable[[str], None] | None = None,
    **kwargs
) -> Image:
    """
    Builds a docker image from a Dockerfile.

    Args:
        client (DockerClient): The docker client instance.
        tag (str): The repository tag to apply to the built image.
        path (str): The path to the directory containing the Dockerfile.
        log_fn (Callable[[str], None] | None, optional): A callback function for logging progress. Defaults to None.
        **kwargs: Arbitrary keyword arguments passed directly to the build process.

    Raises:
        DockerStackException: If the image could not be built.
        RuntimeError: If there is an error during the build process, not including a not found error.

    Returns:
        Image: The built docker image object.
    """
    build_logs = ''
    accumulated_status = ''
    for chunk in client.api.build(tag=tag, path=path, **kwargs):
        _str = chunk.decode('utf-8').rstrip()
        splt_str = _str.split('\n')

        for packet in splt_str:
            msg = json.loads(packet)
            status = msg.get('stream', '')

            if status:
                accumulated_status += status
                if '\n' in accumulated_status:
                    lines = accumulated_status.split('\n')
                    for line in lines[:-1]:
                        if log_fn:
                            log_fn(line)
                        build_logs += line + '\n'
                    accumulated_status = lines[-1]

    try:
        img = client.images.get(tag)
        assert isinstance(img, Image)
        return img
    except docker_errors.NotFound:
        if log_fn:
            log_fn(build_logs.rstrip())
        raise RuntimeError(
            f'Couldn\'t build container {tag} at {path}.'
        )
    except docker_errors.APIError as api_error:
        raise RuntimeError(f'Failed to build the image due to an API error. {api_error}')


def docker_get_running_container(
    client: DockerClient,
    container_name: str
) -> Container | None:
    found = client.containers.list(
        filters={'name': container_name, 'status': 'running'})

    if len(found) == 1:
        return found[0]

    elif len(found) == 0:
        return None

    else:
        raise ValueError(
            f'Expected only one container with name {container_name}')


class UnixSocketConnection(HTTPConnection):
    """A custom HTTPConnection to connect over a Unix Socket."""
    def __init__(self, socket_path: str):
        super().__init__("localhost")
        self.socket_path = socket_path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(self.socket_path)


class UnixSocketConnectionPool(HTTPConnectionPool):
    """A connection pool that uses UnixSocketConnection."""
    def __init__(self, socket_path: str):
        super().__init__("localhost")
        self.socket_path = socket_path

    def _new_conn(self) -> UnixSocketConnection:
        return UnixSocketConnection(self.socket_path)


class UnixSocketAdapter(HTTPAdapter):
    """An adapter for requests to use UnixSocketConnectionPool."""
    def __init__(self, socket_path: str):
        self.socket_path = socket_path
        super().__init__()

    def get_connection(self, url: str, proxies=None) -> UnixSocketConnectionPool:
        return UnixSocketConnectionPool(self.socket_path)


def _parse_docker_log(data: bytes) -> Generator[bytes, None, None]:
    """Parses Docker logs by handling Docker's log protocol."""
    while data:
        header = data[:8]
        _, length = struct.unpack('>BxxxL', header)
        message = data[8:8+length]
        data = data[8+length:]
        yield message


def docker_stream_logs(
    container_name: str,
    socket_path: str = '/var/run/docker.sock',
    timeout: float = 10.0,
    lines: int = 0,
    from_latest: bool = False
) -> Generator[bytes, None, None]:
    """Streams logs from a running Docker container using Unix sockets."""

    session = requests.Session()
    session.mount("http://docker/", UnixSocketAdapter(socket_path))

    url = f"http://docker/containers/{container_name}/logs"
    params = {
        'stdout': '1',
        'stderr': '1',
        'follow': '1',
        'tail': str(lines) if from_latest else 'all'
    }

    response = session.get(url, params=params, stream=True, timeout=timeout)

    data_buffer = b''
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            data_buffer += chunk
            while b'\n' in data_buffer:
                newline_index = data_buffer.index(b'\n') + 1
                log_chunk = data_buffer[:newline_index]
                data_buffer = data_buffer[newline_index:]
                for message in _parse_docker_log(log_chunk):
                    yield message


def docker_open_process(
    client: DockerClient,
    cntr: Container,
    cmd: list[str],
    **kwargs
) -> tuple[str, Iterator[bytes]]:
    """Begin running the command inside the container, return the
    internal docker process id, and a stream for the standard output.
    :param cmd: list of individual string forming the command to execute in
        the testnet container shell.
    :param kwargs: A variable number of key word arguments can be
        provided, as this function uses `exec_create & exec_start docker APIs
        <https://docker-py.readthedocs.io/en/stable/api.html#module-dock
        er.api.exec_api>`_.
    :return: A tuple with the process execution id and the output stream to
        be consumed.
    :rtype: :ref:`typing_exe_stream`
    """
    exec_id = client.api.exec_create(cntr.id, cmd, **kwargs)
    exec_stream = client.api.exec_start(exec_id=exec_id, stream=True)
    return exec_id['Id'], exec_stream


def docker_wait_process(
    client: DockerClient,
    exec_id: str,
    exec_stream: Iterator[bytes],
    logger=None
) -> tuple[int, str]:
    """Collect output from process stream, then inspect process and return
    exitcode.
    :param exec_id: Process execution id provided by docker engine.
    :param exec_stream: Process output stream to be consumed.
    :return: Exitcode and process output.
    :rtype: :ref:`typing_exe_result`
    """
    if logger is None:
        logger = logging.getLogger()

    out = ''
    for chunk in exec_stream:
        msg = chunk.decode('utf-8')
        out += msg

    info = client.api.exec_inspect(exec_id)

    ec = info['ExitCode']
    if ec != 0:
        logger.warning(out.rstrip())

    return ec, out


def docker_move_into(
    client: DockerClient,
    container: str | Container,
    src: str | Path,
    dst: str | Path
):
    tmp_name = random_string(size=32)
    archive_loc = Path(f'/tmp/{tmp_name}.tar.gz').resolve()

    with tarfile.open(archive_loc, mode='w:gz') as archive:
        archive.add(src, recursive=True)

    with open(archive_loc, 'rb') as archive:
        binary_data = archive.read()

    archive_loc.unlink()

    if isinstance(container, Container):
        container = container.id

    client.api.put_archive(container, dst, binary_data)


def docker_move_out(
    container: Container,
    src: str | Path,
    dst: str | Path
):
    tmp_name = random_string(size=32)
    archive_loc = Path(f'/tmp/{tmp_name}.tar.gz').resolve()

    bits, _ = container.get_archive(src, encode_stream=True)

    with open(archive_loc, mode='wb+') as archive:
        for chunk in bits:
            archive.write(chunk)

    extract_path = Path(dst).resolve()

    if extract_path.is_file():
        extract_path = extract_path.parent

    with tarfile.open(archive_loc, 'r') as archive:
        archive.extractall(path=extract_path)

    archive_loc.unlink()



def docker_stop(
    container: Container,
    timeout: float = 30.0,
    poll_time: float = 0.2,
    stop_sequence: list[str] = ['SIGINT', 'SIGTERM']
):

    def _wait_stop_with_timeout():
        try:
            container.reload()
            now = time.time()
            start_time = now
            while (container.status == 'running' and
                    now - start_time < timeout):

                time.sleep(poll_time)
                container.reload()
                now = time.time()

        except docker_errors.NotFound:
            ...


    def _try_stop_with_signal(sig: str) -> bool:
        try:
            container.reload()

            if container.status == 'running':
                # send sig and wait at least timeout or til container stops
                container.kill(signal=sig)
                _wait_stop_with_timeout()

            container.reload()
            return container.status != 'running'

        except docker_errors.NotFound:
            ...

    for sig in stop_sequence:
        if _try_stop_with_signal(sig):
            break
