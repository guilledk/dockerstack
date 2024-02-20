#!/usr/bin/env python3

import time

import requests
import simplejson.errors as sj_errors

from pydantic import BaseModel
from dockerstack.typing import ServiceConfig
from dockerstack.service import DockerService


class IndexPatternDict(BaseModel):
    title: str
    time_field_name: str | None = None


class KibanaDict(ServiceConfig):
    host: str
    patterns: list[IndexPatternDict] = []


class KibanaService(DockerService):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.config: KibanaDict

        self.template_whitelist = [
            'kibana.yml'
        ]

    def prepare(self):
        super().prepare()

        es_service = self.stack.get_service('elastic')

        es_host = es_service.config.protocol + f'://{es_service.ip}:{es_service.ports["http"]}'

        self.environment['ELASTICSEARCH_HOSTS'] = es_host

    def start(self):
        kibana_port = self.config.ports['server']
        kibana_host = self.ip

        for pattern in self.config.patterns:
            self.logger.info(
                f'registering index pattern \'{pattern.title}\'')
            while True:
                try:
                    extra_params = {}

                    if pattern.time_field_name:
                        extra_params['timeFieldName'] = pattern.time_field_name

                    resp = requests.post(
                        f'http://{kibana_host}:{kibana_port}'
                        '/api/index_patterns/index_pattern',
                        headers={'kbn-xsrf': 'true'},
                        json={
                            "index_pattern" : {
                                "title": pattern.title,
                                **extra_params
                            }
                        }).json()

                    self.logger.debug(resp.text)

                    assert resp.status_code == 200

                except requests.exceptions.ConnectionError:
                    self.logger.stack_warning('can\'t reach kibana, retry in 3 sec...')

                except sj_errors.JSONDecodeError:
                    self.logger.stack_info('kibana server not ready, retry in 3 sec...')

                else:
                    break

                time.sleep(3)

            self.logger.stack_info('registered.')

    @property
    def status(self) -> str:
        url = f'http://{self.ip}:{self.ports["server"]}/api/status'
        try:
            response = requests.get(url)
            response.raise_for_status()

        except requests.RequestException:
            return 'unhealthy'

        return 'healthy'
