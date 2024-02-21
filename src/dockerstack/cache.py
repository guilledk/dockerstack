#!/usr/bin/env python3

import os
import json
import shutil
import filelock

from pathlib import Path

from pyunpack import Archive
from zstandard import ZstdDecompressor


class CacheDir:
    def __init__(self, root_dir: Path | str = '~/.cache/dockerstack'):
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)

        self.lock_file = self.root_dir / '.lock'

    def get_path(self, relative_path: str) -> Path:
        '''
        Returns the absolute path to the requested file in the cache,
        following symlinks if necessary.
        '''
        file_path = self.root_dir / relative_path

        result_path = file_path
        if file_path.is_symlink():
            result_path = file_path.resolve(strict=True)

        res_dir = result_path.parent

        if not res_dir.exists():
            res_dir.mkdir(parents=True)

        return result_path

    def file_exists(self, relative_path: str) -> bool:
        '''
        Checks if a file exists in the cache (following symlinks).
        '''
        return self.get_path(relative_path).exists()

    def store_file(self, file_path: Path, relative_path: str):
        '''
        Stores a file in the cache under the given relative path.
        If the path already exists, it will be overwritten.
        '''
        target_path = self.get_path(relative_path)
        if target_path.is_dir():
            shutil.copytree(file_path, target_path, dirs_exist_ok=True)
        else:
            shutil.copy(file_path, target_path)

    def store_json(self, jdata: dict, relative_path: str):
        '''
        Stores a json dict in the cache under the given relative path.
        If the path already exists, it will be overwritten.
        '''
        target_path = self.get_path(relative_path)

        if target_path.exists():
            if target_path.is_dir():
                raise FileExistsError(f'{target_path} exists and is a dir?')

            else:
                target_path.unlink()

        with open(target_path, 'w+') as file:
            json.dump(jdata, file, indent=4)

    def create_alias(self, src: str, dst: str):
        '''
        Creates a symlink in the cache directory, pointing from dst to src.
        If the dst symlink already exists, it will be replaced.
        '''
        src_path = self.get_path(src)
        dst_path = self.get_path(dst)

        if dst_path.exists() or dst_path.is_symlink():
            dst_path.unlink()

        os.symlink(src_path, dst_path)

    def retrieve_file(self, relative_path: str) -> Path:
        '''
        Retrieves a file from the cache, following symlinks if necessary.
        Raises FileNotFoundError if the file does not exist.
        '''
        file_path = self.get_path(relative_path)
        if not file_path.exists():
            raise FileNotFoundError(f'No cached file found at {relative_path}')

        return file_path

    def retrieve_json(self, relative_path: str) -> dict:
        '''
        Retrieves a json file from the cache, following symlinks if necessary.

        Raises FileNotFoundError if the file does not exist.
        '''
        file_path = self.get_path(relative_path)
        if not file_path.exists():
            raise FileNotFoundError(f'No cached file found at {relative_path}')

        with open(file_path, 'r') as file:
            return json.load(file)

    def extract_file(self, relative_path: str) -> list[str]:
        '''
        Extracts a compressed file located at relative_path and tracks all the new files created during the extraction.
        Only allows one extraction at a time using a file system lock.
        Returns a list of Paths to the newly created files.
        '''
        with filelock.FileLock(str(self.lock_file)):
            # take a snapshot of existing files
            before_files = {f for f in self.root_dir.glob('**/*') if f.is_file()}

            # perform extraction
            target_path = self.get_path(relative_path)
            extraction_dir = target_path.parent

            if target_path.suffix in ['.zst', '.zstd']:
                with (
                    open(target_path, 'rb') as src_file,
                    open(extraction_dir / target_path.stem, 'wb') as dst_file,
                    ZstdDecompressor().stream_reader(src_file) as reader
                ):
                    shutil.copyfileobj(reader, dst_file)

            else:
                Archive(str(target_path)).extractall(str(extraction_dir))

            # take a snapshot of files after extraction
            after_files = {f for f in self.root_dir.glob('**/*') if f.is_file()}

            # determine the new files by comparing snapshots
            new_files = after_files - before_files


            relatives = {
                f.relative_to(extraction_dir).parts[0]
                for f in new_files
            }

            return list(relatives)
