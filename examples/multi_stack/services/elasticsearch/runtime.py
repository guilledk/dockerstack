#!/usr/bin/env python3

import requests

from dockerstack.typing import ServiceConfig
from dockerstack.service import DockerService


class ElasticsearchConfig(ServiceConfig):
    protocol: str
    cluster_name: str = 'es-cluster'
    node_name: str = 'es-example'


class ElasticsearchService(DockerService):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.config: ElasticsearchConfig

        self.node_url = f'{self.config.protocol}://{self.ip}:{self.ports["http"]}'

        self.template_whitelist = [
            'elasticsearch.yml'
        ]

    @property
    def status(self) -> str:
        try:
            response = requests.get(self.node_url + '/_cluster/health')
            response.raise_for_status()

        except requests.RequestException:
            return 'unhealthy'

        return 'healthy'
