#!/usr/bin/env python3

import requests

from dockerstack.typing import CommonDict
from dockerstack.service import DockerService


class ElasticsearchDict(CommonDict):
    protocol: str
    cluster_name: str = 'es-cluster'
    node_name: str = 'es-example'


class ElasticsearchService(DockerService):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.config: ElasticsearchDict

        self.template_whitelist = [
            'elasticsearch.yml'
        ]

    @property
    def status(self) -> str:
        url = f'{self.config.protocol}://{self.ip}:{self.ports["http"]}/_cluster/health'
        try:
            response = requests.get(url)
            response.raise_for_status()

        except requests.RequestException:
            return 'unhealthy'

        return 'healthy'
