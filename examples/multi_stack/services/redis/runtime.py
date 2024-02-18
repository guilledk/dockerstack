#!/usr/bin/env python3

import socket

from dockerstack.typing import CommonDict
from dockerstack.service import DockerService


class RedisDict(CommonDict):
    host: str


class RedisService(DockerService):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.config: RedisDict

        self.template_whitelist = [
            'redis.conf'
        ]

    @property
    def status(self) -> str:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(4)
                s.connect((self.ip, self.ports['bind']))

                s.sendall(b'PING\r\n')
                if not s.recv(1024).startswith(b'+PONG'):
                    raise AssertionError

                s.sendall(b'info replication\r\n')
                replication_info = s.recv(4096)
                if b'role:master' not in replication_info:
                    raise AssertionError

                s.sendall(b'QUIT\r\n')
                if not s.recv(1024).startswith(b'+OK'):
                    raise  AssertionError

            return 'healthy'

        except:
            return 'unhealthy'
