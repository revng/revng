#!/usr/bin/env python3

#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import os
import struct
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from tempfile import mkstemp
from typing import IO

import pefile
import requests
from elftools.common.exceptions import ELFError
from elftools.elf.elffile import ELFFile
from elftools.elf.sections import NoteSection
from urllib3.util import Retry


class Logger:
    def __init__(self):
        self.verbose = False

    def log_error(self, message):
        sys.stderr.write(message + "\n")

    def log(self, message):
        if self.verbose:
            sys.stderr.write(message + "\n")


logger = Logger()
log = logger.log
log_error = logger.log_error


session = requests.Session()
adapter = requests.adapters.HTTPAdapter(
    max_retries=Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=[
            408,  # Request timeout
            429,  # Too Many Requests
            500,  # Internal Server Error
            502,  # Bad Gateway
            503,  # Service Unavailable
            504,  # Gateway Timeout
        ],
    )
)
session.mount("http://", adapter)
session.mount("https://", adapter)


def download_file(url, local_filename):
    log(f"Downloading {local_filename}")
    try:
        with session.get(url, stream=True) as request:
            if request.status_code == 200:
                down_fd, download_name = mkstemp(dir=os.path.dirname(local_filename))
                with open(down_fd, "wb") as debug_file:
                    for chunk in request.iter_content(chunk_size=64 * 1024):
                        debug_file.write(chunk)
                log("Downloaded")
                os.replace(download_name, local_filename)
                return True
            elif request.status_code == 404:
                log("URL was not found")
            else:
                log(f"URL returned status code {request.status_code}")
    except requests.RequestException as e:
        log(f"Exception while making request: {e}")
    return False


def is_elf(file: IO[bytes]):
    file.seek(0, os.SEEK_SET)
    return file.read(4) == b"\x7fELF"


def parse_build_id(file):
    for section in file.iter_sections():
        if not isinstance(section, NoteSection):
            continue
        for note in section.iter_notes():
            desc = note["n_desc"]
            if note["n_type"] != "NT_GNU_BUILD_ID":
                continue
            return desc
    return None


def fetch_dwarf(file: IO[bytes], urls: tuple[str, ...], output: Path):
    try:
        elf_file = ELFFile(file)
        log("Looking for Debugging Information for an ELF")
        build_id = parse_build_id(elf_file)
    except ELFError as elf_error:
        log_error(str(elf_error))
        return None

    if build_id is None:
        # If we cannot parse the build id, we cannot fetch debug info.
        log("Parsing build-id failed")
        return None

    log("BUILD-ID: " + build_id)

    # Find debug info on the web as `debuginfod` does.
    # Ensure that we have created `$cache_directory/debug-symbols/elf` directory.
    path_to_revng_elf_debug_data = output / "debug-symbols" / "elf"
    path_to_revng_elf_debug_data.mkdir(parents=True, exist_ok=True)

    directory_path_to_download = path_to_revng_elf_debug_data / build_id
    directory_path_to_download.mkdir(exist_ok=True)

    debug_file_to_download = directory_path_to_download / "debug"
    if debug_file_to_download.exists():
        log("Already downloaded debug file from web")
        return debug_file_to_download

    log("Trying to find the debug info on the web")
    for url in urls:
        debug_info_url = f"{url}/buildid/{build_id}/debuginfo"
        log(f"Trying to download from {debug_info_url}")
        if download_file(debug_info_url, str(debug_file_to_download)):
            return debug_file_to_download

    return None


def is_pe(file: IO[bytes]):
    file.seek(0, os.SEEK_SET)
    mz_header = file.read(2)
    if mz_header != b"MZ":
        return False
    file.seek(0x3C, os.SEEK_SET)  # 0x3C contains PE offset
    pe_offset = struct.unpack("<i", file.read(4))[0]  # Offset is encoded as a little-endian int
    file.seek(pe_offset, os.SEEK_SET)
    return file.read(4) == b"PE\x00\x00"


def fetch_pdb(file_path: Path, urls: tuple[str, ...], output: Path) -> Path | None:
    log("Fetching Debugging Information for a PE/COFF")

    # Parse the file.
    try:
        pe = pefile.PE(file_path)
    except pefile.PEFormatError as exception:
        log_error("Unable to parse the input file as PE/COFF.")
        log_error(str(exception))
        return None

    # Ensure that we have created `$cache_directory/debug-symbols/pe` directory.
    path_to_revng_pe_debug_data = output / "debug-symbols" / "pe"
    path_to_revng_pe_debug_data.mkdir(parents=True, exist_ok=True)

    for e in pe.DIRECTORY_ENTRY_DEBUG:
        if isinstance(e.entry, pefile.Structure):
            pdb_file_name_raw = e.entry.PdbFileName.split(b"\x00")[0].decode("utf-8")
            pdb_file_name = PureWindowsPath(pdb_file_name_raw).name

            guid = f"{e.entry.Signature_Data1:08x}"
            guid += f"{e.entry.Signature_Data2:04x}"
            guid += f"{e.entry.Signature_Data3:04x}"
            guid += f"{e.entry.Signature_Data4:02x}"
            guid += f"{e.entry.Signature_Data5:02x}"
            guid += f'{int.from_bytes(e.entry.Signature_Data6, byteorder="big"):012x}'
            pdb_id = f"{guid.upper()}{e.entry.Age:x}".lower()

            log("PDBID: " + pdb_id)

            path_to_download = path_to_revng_pe_debug_data / pdb_id
            path_to_download.mkdir(exist_ok=True)

            pdb_file_to_download = path_to_download / pdb_file_name
            if pdb_file_to_download.exists():
                log("Already downloaded debug file from web")
                return pdb_file_to_download

            for symbol_server_url in urls:
                symbol_server_url = symbol_server_url.rstrip("/")
                pdb_url = f"{symbol_server_url}/{pdb_file_name}/{pdb_id}/{pdb_file_name}"
                log("Trying to download PDB from URL: " + pdb_url)
                if download_file(pdb_url, str(pdb_file_to_download)):
                    return pdb_file_to_download

            break

    return None


@dataclass
class Options:
    elf_servers: tuple[str, ...] = ("https://debuginfo-cache.rev.ng/elf",)
    pe_servers: tuple[str, ...] = ("https://debuginfo-cache.rev.ng/pe",)
    output_dir: Path = Path.cwd()


def fetch_debuginfo(input_path: str, options: Options) -> Path | None:
    path = Path(input_path)
    result: Path | None = None
    with path.open(mode="rb") as f:
        # Check if it is an ELF.
        if is_elf(f):
            result = fetch_dwarf(f, options.elf_servers, options.output_dir)
        elif is_pe(f):
            # TODO: handle _NT_SYMBOL_PATH and _NT_ALT_SYMBOL_PATH
            # https://learn.microsoft.com/en-us/windows-hardware/drivers/debugger/symbol-path
            result = fetch_pdb(path, options.pe_servers, options.output_dir)

    return result


# While the logic in this file can be accessed through the
# `revng model fetch-debuginfo` command, this file is intended to be
# additionally used as a standalone script, hence the presence of a `main`
# function here.
def main():
    parser = ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("-o", "--output", help="Output directory")
    parser.add_argument(
        "--elf-server", action="append", help="ELF server to use (can be specified multiple times)"
    )
    parser.add_argument(
        "--pe-server", action="append", help="PE server to use (can be specified multiple times)"
    )
    parser.add_argument("input", help="Input file")
    args = parser.parse_args()
    if args.verbose:
        logger.verbose = True

    options = Options()
    if args.output is not None:
        options.output_dir = Path(args.output)
    if args.elf_server is not None:
        options.elf_servers = tuple(args.elf_server)
    if args.pe_server is not None:
        options.pe_servers = tuple(args.pe_server)

    result = fetch_debuginfo(args.input, options)
    return 0 if result is not None else 1


if __name__ == "__main__":
    sys.exit(main())
