#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from pathlib import PureWindowsPath

import pefile

from revng.cli.support import log_error

from .common import cache_directory, download_file, log


def fetch_pdb(file_path, urls):
    log("Fetching Debugging Information for a PE/COFF")

    # Parse the file.
    try:
        pe = pefile.PE(file_path)
    except pefile.PEFormatError as exception:
        log_error("Unable to parse the input file as PE/COFF.")
        log_error(str(exception))
        return None

    # Ensure that we have created `$cache_directory/debug-symbols/pe` directory.
    path_to_revng_pe_debug_data = cache_directory() / "debug-symbols" / "pe"
    path_to_revng_pe_debug_data.mkdir(parents=True, exist_ok=True)

    for e in pe.DIRECTORY_ENTRY_DEBUG:
        if isinstance(e.entry, pefile.Structure):
            pdb_file_name_raw = e.entry.PdbFileName.decode("utf-8").replace("\x00", "")
            pdb_file_name = PureWindowsPath(pdb_file_name_raw).name

            guid = f"{e.entry.Signature_Data1:08x}"
            guid += f"{e.entry.Signature_Data2:04x}"
            guid += f"{e.entry.Signature_Data3:04x}"
            guid += f"{e.entry.Signature_Data4:02x}"
            guid += f"{e.entry.Signature_Data5:02x}"
            guid += f'{int.from_bytes(e.entry.Signature_Data6, byteorder="big"):012x}'
            pdb_id = f"{guid.upper()}{e.entry.Age:x}"

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
