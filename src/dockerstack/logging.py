#!/usr/bin/env python3

import logging


class Formatter(logging.Formatter):

    COLORS = {
        'STACK_INFO': '\033[94m',     # Blue
        'STACK_WARNING': '\033[93m',  # Yellow
        'STACK_ERROR': '\033[91m',    # Red
    }

    RESET = '\033[0m'

    _DISPLAY = {
        'STACK_INFO': 'info',
        'STACK_WARNING': 'warning',
        'STACK_ERROR': 'error'
    }

    def __init__(self):
        super().__init__('%(asctime)s %(funcName)-15s:%(lineno)-5d %(message)s', '%Y-%m-%dT%H:%M:%S%z')

    def format(self, record):
        if len(record.funcName) < 20:
            padding_length = 20 - len(record.funcName)
            record.funcName = record.funcName[:20] + ' ' * padding_length
        else:
            record.funcName = record.funcName[:20]
        log_message = super().format(record)
        levelname = record.levelname
        if levelname not in self.COLORS:
            return log_message
        return f"{self.COLORS[levelname]}{self._DISPLAY[levelname]}{self.RESET} : {log_message}"


# Define custom log levels
STACK_INFO_NUM = 25
STACK_WARNING_NUM = 35
STACK_ERROR_NUM = 45


logging.addLevelName(STACK_INFO_NUM, 'STACK_INFO')
logging.addLevelName(STACK_WARNING_NUM, 'STACK_WARNING')
logging.addLevelName(STACK_ERROR_NUM, 'STACK_ERROR')


class DockerStackLogger(logging.Logger):

    def info(self, message, *args, **kwargs):
        if self.isEnabledFor(STACK_INFO_NUM):
            self._log(STACK_INFO_NUM, message, args, **kwargs)

    def warning(self, message, *args, **kwargs):
        if self.isEnabledFor(STACK_WARNING_NUM):
            self._log(STACK_WARNING_NUM, message, args, **kwargs)

    def error(self, message, *args, **kwargs):
        if self.isEnabledFor(STACK_ERROR_NUM):
            self._log(STACK_ERROR_NUM, message, args, **kwargs)

    def stack_info(self, message, *args, **kwargs):
        if self.isEnabledFor(STACK_INFO_NUM):
            self._log(STACK_INFO_NUM, message, args, **kwargs)

    def stack_warning(self, message, *args, **kwargs):
        if self.isEnabledFor(STACK_WARNING_NUM):
            self._log(STACK_WARNING_NUM, message, args, **kwargs)

    def stack_error(self, message, *args, **kwargs):
        if self.isEnabledFor(STACK_ERROR_NUM):
            self._log(STACK_ERROR_NUM, message, args, **kwargs)


def get_stack_logger(level: str) -> DockerStackLogger:
    # Set the logger class to DockerStackLogger
    logging.setLoggerClass(DockerStackLogger)

    # Check if root logger is already initialized
    if logging.root.handlers:
        logging.root.handlers = []

    # Directly set the class of the root logger
    logging.root.__class__ = DockerStackLogger

    # Setup the root logger
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    handler.setFormatter(Formatter())
    logger.addHandler(handler)
    logger.setLevel(level.upper())

    logger.propagate = False
    logging.getLogger().addHandler(logging.NullHandler())

    return logger
