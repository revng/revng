#!/usr/bin/env python3
#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import os
import shutil
import sys

import pefile
import requests

from revng.cli.commands_registry import Command, CommandsRegistry, Options
from revng.cli.support import log_error

microsoft_symbol_server_url = "http://msdl.microsoft.com/download/symbols/"


class DownloadPDBCommand(Command):
    def __init__(self):
        super().__init__(("model", "download-pdb"), "Download missing PDBs for CRTs.")

    def register_arguments(self, parser):
        parser.add_argument("input", help="The input file.")
        parser.add_argument("output", help="The output file.", nargs="?", default="/dev/stdout")
        parser.add_argument(
            "urls",
            help="List of URLs where to try finding the PDB files.",
            nargs="*",
            default=[microsoft_symbol_server_url],
        )

    def log(self, message):
        if self.verbose:
            sys.stderr.write(message + "\n")

    def download_pdb_file(self, url, local_filename):
        with requests.get(url, stream=True) as r, open(local_filename, "wb") as f:
            if r.status_code == 200:
                shutil.copyfileobj(r.raw, f)
                self.log("Downloaded.")
                return True
            else:
                self.log("URL was not found.")
                return False

        return False

    def run(self, options: Options):
        args = options.parsed_args
        self.verbose = args.verbose

        if not os.path.exists(args.input):
            log_error("Could not find " + args.input)
            return 1

        # Parse the file.
        try:
            pe = pefile.PE(args.input)
        except Exception as exception:
            log_error("Unable to parse the input file.")
            log_error(str(exception))
            return 1

        for e in pe.DIRECTORY_ENTRY_DEBUG:
            if not isinstance(e.entry, pefile.Structure):
                continue

            pdb_file_name = e.entry.PdbFileName.decode("utf-8")
            pdb_file_name = pdb_file_name.replace("\x00", "")
            # A hash for the PDB file.
            pdb_id = (
                str(hex(e.entry.Signature_Data1)[2:])
                + str(hex(e.entry.Signature_Data2)[2:])
                + str(hex(e.entry.Signature_Data3)[2:])
                + e.entry.Signature_Data4.hex()
                + str(hex(e.entry.Age)[2:])
            )
            self.log("PDBID: " + pdb_id)

            for symbol_server_url in args.urls:
                pdb_url = symbol_server_url + pdb_file_name + "/" + pdb_id + "/" + pdb_file_name
                self.log("URL: " + pdb_url)

                if self.download_pdb_file(pdb_url, args.output) is True:
                    return 0
            break

        return 0


def setup(commands_registry: CommandsRegistry):
    commands_registry.register_command(DownloadPDBCommand())
