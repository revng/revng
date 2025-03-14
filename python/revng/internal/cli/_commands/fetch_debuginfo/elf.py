#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from elftools.elf.elffile import ELFFile
from elftools.elf.sections import NoteSection

from revng.internal.support import cache_directory

from .common import download_file, log


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


def fetch_dwarf(file: ELFFile, file_path, urls):
    log("Looking for Debugging Information for an ELF")
    build_id = parse_build_id(file)
    if build_id is None:
        # If we cannot parse the build id, we cannot fetch debug info.
        log("Parsing build-id failed")
        return None

    log("BUILD-ID: " + build_id)

    # Find debug info on the web as `debuginfod` does.
    # Ensure that we have created `$cache_directory/debug-symbols/elf` directory.
    path_to_revng_elf_debug_data = cache_directory() / "debug-symbols" / "elf"
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
