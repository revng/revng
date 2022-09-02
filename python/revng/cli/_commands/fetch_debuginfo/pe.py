#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from pathlib import Path, PureWindowsPath

import pefile
from xdg import xdg_cache_home

from revng.cli.support import log_error

from .common import download_file, log


def fetch_pdb(file: pefile, file_path, urls):
    log("Fetching Debugging Information for a PE/COFF")

    # Parse the file.
    try:
        pe = pefile.PE(file_path)
    except Exception as exception:
        log_error("Unable to parse the input file.")
        log_error(str(exception))
        return None

    # Ensure that we have created `$xdg_cache_home/revng/debug-symbols/pe` directory.
    path_to_revng_data = Path(str(xdg_cache_home()) + "/revng/")
    if not path_to_revng_data.exists():
        path_to_revng_data.mkdir()
    path_to_revng_debug_data = path_to_revng_data / "debug-symbols/"
    if not path_to_revng_debug_data.exists():
        path_to_revng_debug_data.mkdir()
    path_to_revng_pe_debug_data = path_to_revng_debug_data / "pe/"
    if not path_to_revng_pe_debug_data.exists():
        path_to_revng_pe_debug_data.mkdir()

    for e in pe.DIRECTORY_ENTRY_DEBUG:
        if isinstance(e.entry, pefile.Structure):
            pdb_file_name = e.entry.PdbFileName.decode("utf-8")
            pdb_file_name = pdb_file_name.replace("\x00", "")

            full_path = PureWindowsPath(pdb_file_name)
            pdb_file_name = full_path.name

            # Parse the hash that identifies the PDB file.
            def pdb_id_helper(data):
                return hex(data)[2:]

            pdb_id = (
                pdb_id_helper(e.entry.Signature_Data1)
                + pdb_id_helper(e.entry.Signature_Data2)
                + pdb_id_helper(e.entry.Signature_Data3)
                + e.entry.Signature_Data4.hex()
                + pdb_id_helper(e.entry.Age)
            )

            log("PDBID: " + pdb_id)

            path_to_download = path_to_revng_pe_debug_data / pdb_id
            if not path_to_download.exists():
                path_to_download.mkdir()

            pdb_file_to_download = path_to_download / pdb_file_name
            if pdb_file_to_download.exists():
                log("Already downloaded debug file from web")
                return pdb_file_to_download

            for symbol_server_url in urls:
                pdb_url = symbol_server_url + pdb_file_name + "/" + pdb_id + "/" + pdb_file_name
                log("Trying to download PDB from URL: " + pdb_url)
                if download_file(pdb_url, str(pdb_file_to_download)):
                    return pdb_file_to_download

            break

    return None
