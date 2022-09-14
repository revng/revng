#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from pathlib import Path

from revng.cli.commands_registry import Command, Options
from revng.cli.revng import run_revng_command

import json
import os
import marko
import sys
import argparse
import subprocess

from tempfile import NamedTemporaryFile
from collections import defaultdict


_verbose = False

def log(message):
    if _verbose:
        sys.stderr.write(message + "\n")


def only(list):
    assert len(list) == 1
    return list[0]


class PythonDoctest:
    def __init__(self):
        self.extension = ".py"

    def command(self, path):
        return ["python", "-c", "import doctest; import sys; doctest.testfile(sys.argv[-1])", path]

    def process(self, example):
        return example + "\n"


class BashDoctest:
    def __init__(self):
        self.extension = ".sh"

    def command(self, path):
        return ["docshtest", path]

    def process(self, example):
        return "    " + example.replace("\n", "\n    ") + "\n"


def escape_js(string):
    assert type(string) is str
    return json.dumps(string)


def emit_assertion(expected):
    return f"if (JSON.stringify(last) !== {escape_js(expected[:-1])})\n  process.exit(1);\n"

# WIP
class TypeScriptDoctest:
    def __init__(self):
        self.extension = ".ts"

    def command(self, path):
        return ["bash", "-c", f"""./node_modules/.bin/tsc "{path}" && node "{path.replace(".ts", ".js")}" && rm "{path.replace(".ts", ".js")}" """]

    def process(self, example):
        output = "console.log = (x) => { return x; };\nlet last;\n"

        last_output = ""
        lines = list(map(str.strip, example.split("\n")))[:-1]
        for line, next_line in zip(lines, lines[1:] + ["> "]):
            is_command = line.startswith("> ")
            next_is_command = next_line.startswith("> ")

            if is_command:
                if last_output:
                    output += emit_assertion(last_output)
                    last_output = ""

                line = line[2:]
                if next_is_command:
                    output += f"{line}\n"
                else:
                    output += f"last = {line}\n"

            else:
                last_output += line + "\n"

        if last_output:
            output += emit_assertion(last_output)

        return output



def run(arguments):
    log(f"Running {str(arguments)}")
    return subprocess.run(arguments, check=True)


class TestDocsCommand(Command):
    def __init__(self):
        super().__init__(("test-docs",), "revng documentation tester")


    def register_arguments(self, parser):
        parser.add_argument(
            "--verbose", action="store_true", help="Be verbose."
        )

        parser.add_argument("files",
                            metavar="FILE",
                            nargs="*",
                            help="files to test.")

    def run(self, options: Options):
        args = options.parsed_args

        global _verbose
        _verbose = options.verbose

        if args.files:
            files = [Path(path) for path in args.files]
        else:
            files = []
            for search_prefix in options.search_prefixes:
                docs_path = Path(search_prefix) / "share" / "doc" / "revng"
                reading_order_path = docs_path / "READING-ORDER"
                if reading_order_path.exists():
                    files += [docs_path / docfile_path
                              for docfile_path
                              in reading_order_path.read_text().split("\n")
                              if docfile_path]


        handlers = {
            "python": PythonDoctest(),
            "bash": BashDoctest(),
            # "typescript": TypeScriptDoctest()
        }

        scripts = defaultdict(str)

        for path in files:
            document = marko.parse(path.read_text())

            for block in document.children:
                if type(block) is not marko.block.FencedCode:
                    continue

                if block.lang not in handlers:
                    continue

                rawtext = only(block.children)
                assert type(rawtext) is marko.inline.RawText
                text = rawtext.children
                assert type(text) is str

                scripts[block.lang] += handlers[block.lang].process(text)

        for language, script in scripts.items():
            handler = handlers[language]

            with NamedTemporaryFile("w",
                                    suffix=handler.extension,
                                    dir=os.getcwd()) as temporary_file:
                temporary_file.write(script)
                temporary_file.flush()
                run(handler.command(os.path.basename(temporary_file.name)))
