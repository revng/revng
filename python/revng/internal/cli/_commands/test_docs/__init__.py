#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import abc
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Mapping

import click
import marko

from revng.internal.cli.common import CommandRegistry, cli_logger

SCRIPT_DIR = Path(__file__).parent.resolve()


def run(working_directory: Path, arguments):
    cli_logger.debug_log(f"Running {shlex.join(arguments)}")
    process = subprocess.run(
        arguments, cwd=working_directory, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )

    if process.returncode != 0:
        sys.stderr.write(f"The following program failed: {shlex.join(arguments)}\n")
        sys.stderr.flush()
        sys.stderr.buffer.write(process.stdout)
        sys.stderr.flush()
        sys.stderr.write(f"Process exited with code {process.returncode}\n")
        sys.stderr.write("Re-run with --verbose for further info\n")
        sys.exit(1)


class Doctest(abc.ABC):
    @abc.abstractmethod
    def run(self, working_directory: Path):
        pass

    @abc.abstractmethod
    def process(self, code: str, extra: str = ""):
        pass

    @staticmethod
    def _parse_extra(extra: str) -> dict[str, str | None]:
        if extra.strip() == "":
            return {}

        result: dict[str, str | None] = {}
        for part in re.split(r"\s+", extra):
            assert part.count("=") in (0, 1)
            if part.count("=") == 0:
                result[part] = None
            else:
                key, value = part.split("=", 1)
                result[key] = value
        return result


class PythonDoctest(Doctest):
    def __init__(self):
        self.script = ""

    def run(self, working_directory: Path):
        if not self.script:
            return

        self.script = f'"""{self.script}"""'
        cli_logger.debug_log(
            f"Running the following Python script:\n{textwrap.indent(self.script, '    ')}\n"
        )

        script_path = working_directory / "run.py"
        script_path.write_text(self.script)
        run(working_directory, ["python", str(SCRIPT_DIR / "doctest_runner.py"), "-v", "run.py"])
        self.script = ""

    def process(self, code: str, extra: str = ""):
        extra_decoded = self._parse_extra(extra)
        if "ignoreoutput" in extra_decoded:
            ignore_output = extra_decoded["ignoreoutput"]
            split_lines = code.splitlines()

            if ignore_output is None:
                lines_to_ignore_output = list(range(1, len(split_lines) + 1))
            else:
                lines_to_ignore_output = [int(x) for x in ignore_output.split(",")]

            for index, line in enumerate(split_lines):
                if index + 1 in lines_to_ignore_output:
                    self.script += line + "  # doctest: +IGNORE_OUTPUT\n"
                else:
                    self.script += line + "\n"
            self.script += "\n"
        else:
            self.script += code + "\n"


class BashDoctest(Doctest):
    def __init__(self):
        self.script = ""
        self.expected_output = ""

    def run(self, working_directory: Path):
        if not self.script and not self.expected_output:
            return

        self.script = self.script.strip() + "\n"
        script_path = working_directory / "run.sh"
        script_path.write_text(self.script)

        self.expected_output = self.expected_output.strip() + "\n"
        expected_output_path = working_directory / "expected_output.log"
        expected_output_path.write_text(self.expected_output)

        cli_logger.debug_log(
            f"Running the following bash script:\n{textwrap.indent(self.script, '    ')}\n"
        )
        cli_logger.debug_log(
            f"Expected output is:\n{textwrap.indent(self.expected_output, '    ')}\n"
        )

        run(
            working_directory,
            [
                "bash",
                "-c",
                "bash -euo pipefail ./run.sh |& tee output.log"
                + " && diff -Bwu output.log expected_output.log",
            ],
        )
        self.script = ""
        self.expected_output = ""

    @staticmethod
    def _heredoc_terminator(command: str) -> str | None:
        """If `command` ends with a here-document redirection (`<< WORD`,
        `<<'WORD'`, `<<-WORD`), return its terminator word, so the here-document
        body can be treated as part of the command rather than expected output.
        """
        match = re.search(r"<<-?\s*(['\"]?)([A-Za-z_]\w*)\1\s*$", command)
        return match.group(2) if match else None

    def process(self, code, extra=""):
        next_line_is_command = False
        heredoc_terminator = None
        silent = "silent" in extra
        match = re.match('.*ignore="([^"]*)".*', extra)
        ignore_regexp = None
        if match:
            ignore_regexp = match.groups()[0]
            self.script += "( "

        if silent:
            self.script += "( "

        for line in code.split("\n"):
            if heredoc_terminator is not None:
                self.script += line + "\n"
                if line.strip() == heredoc_terminator:
                    heredoc_terminator = None
            elif line.startswith("$ "):
                self.script += line[2:] + "\n"
                heredoc_terminator = self._heredoc_terminator(line[2:])
                if heredoc_terminator is None and line.endswith("\\"):
                    next_line_is_command = True
            elif next_line_is_command:
                self.script += line + "\n"
                heredoc_terminator = self._heredoc_terminator(line)
                if heredoc_terminator is None:
                    next_line_is_command = line.endswith("\\")
            else:
                if not ignore_regexp:
                    self.expected_output += line + "\n"
                else:
                    self.expected_output += re.sub(ignore_regexp, "IGNORED", line) + "\n"

        if silent:
            self.script += " ) >& /dev/null\n"

        if ignore_regexp:
            # TODO: in the future we could replace only certain parts of the regexp using named
            #       groups: `struct_(?P<ignore>[0-9]+)` would turn `a struct_1234 b` into
            #       `a struct_IGNORED b` instead of `a IGNORED b`, as this does.
            filter_script = """
import re
import sys
pattern = re.compile(sys.argv[1])
sys.stdout.writelines(pattern.sub("IGNORED", line) for line in sys.stdin)
"""
            self.script += (
                f" ) |& python3 -c {shlex.quote(filter_script)} {shlex.quote(ignore_regexp)}\n"
            )


class TypeScriptDoctest(Doctest):
    def __init__(self):
        self.script = ""

    def run(self, working_directory: Path):
        if not self.script:
            return

        self.script = self.script.strip() + "\n"
        cli_logger.debug_log(
            f"Running the following TypeScript script:\n{textwrap.indent(self.script, '    ')}"
        )
        script_path = working_directory / "run.ts"
        script_path.write_text(self.script)
        # If `tsc` and `revng-model` are already provided by the
        # environment (e.g. nix's `revng-test-node-env`), skip
        # `npm install`. tsc's module resolution still wants a
        # local `node_modules/`, so symlink the env's tree into
        # the working directory — that gives both `revng-model`
        # and `@types/node` to the type checker.
        # Otherwise fall back to the original online install so
        # this still works under orchestra and ad-hoc invocations.
        tsc_path = shutil.which("tsc")
        if tsc_path is not None:
            env_node_modules = Path(tsc_path).resolve().parent.parent / "lib" / "node_modules"
            if env_node_modules.is_dir():
                link = working_directory / "node_modules"
                if not link.exists():
                    link.symlink_to(env_node_modules)
            cmd = "tsc run.ts && node run.js"
        else:
            cmd = (
                "npm install typescript revng-model @types/node"
                + " && ./node_modules/.bin/tsc run.ts"
                + " && node run.js"
            )
        run(working_directory, ["bash", "-c", cmd])
        self.script = ""

    @staticmethod
    def escape_js(string):
        assert type(string) is str
        return json.dumps(string)

    @classmethod
    def emit_assertion(cls, expected):
        return f"if (JSON.stringify(last) !== {cls.escape_js(expected[:-1])})\n  process.exit(1);\n"

    def process(self, code, extra=""):
        output = "console.log = (x) => { return x; };\nlet last;\n"

        last_output = ""
        lines = list(map(str.strip, code.split("\n")))[:-1]
        for line, next_line in zip(lines, lines[1:] + ["> "]):
            is_command = line.startswith("> ")
            next_is_command = next_line.startswith("> ")

            if is_command:
                if last_output:
                    output += self.emit_assertion(last_output)
                    last_output = ""

                line = line[2:]
                if next_is_command:
                    output += f"{line}\n"
                else:
                    output += f"last = {line}\n"

            else:
                last_output += line + "\n"

        if last_output:
            output += self.emit_assertion(last_output)

        self.script = output


def only(entries):
    assert len(entries) == 1
    return entries[0]


def handle_file(path: Path):
    cli_logger.debug_log(f"Processing {str(path)}")

    handler_types = {
        "python": PythonDoctest,
        "bash": BashDoctest,
        "typescript": TypeScriptDoctest,
    }

    handlers: Mapping[str, Doctest] = {
        language: constructor() for language, constructor in handler_types.items()  # type: ignore
    }

    document = marko.parse(path.read_text())

    for block in document.children:
        if type(block) is not marko.block.FencedCode:
            continue

        block.lang = block.lang.strip("{").strip("}")
        block.extra = block.extra.strip("{").strip("}")

        if "notest" in block.extra:
            cli_logger.debug_log(
                f"Ignoring fenced code of type {block.lang}"
                + f' due to `notest` in "{block.extra}"'
            )
            continue

        if "noorchestra" in block.extra:
            cli_logger.debug_log(
                f'Disabling orchestra environment due to `noorchestra` in "{block.extra}"'
            )
            handlers["bash"].process(
                "$ unset ORCHESTRA_DOTDIR;"
                + " unset ORCHESTRA_ROOT;"
                + " unset ORCHESTRA_NODE_CACHE;"
                + " export ORCHESTRA_DOTDIR ORCHESTRA_ROOT ORCHESTRA_NODE_CACHE;"
            )

        rawtext = only(block.children)
        assert type(rawtext) is marko.inline.RawText
        text = rawtext.children
        assert type(text) is str

        match = re.match(r"title=([^ ]*\.[^ ]*)", block.extra)
        if match:
            append_to = match.groups()[0]
            cli_logger.debug_log(f"Appending to {append_to}")
            if text.endswith("\n"):
                text = text[:-1]
            for line in text.split("\n"):
                handlers["bash"].process("$ echo " + shlex.quote(line) + " >> " + append_to)
        elif block.lang == "diff":
            cli_logger.debug_log("Applying diff")
            handlers["bash"].process("""$ rm -f patch.patch""")
            for line in text.split("\n"):
                handlers["bash"].process("$ echo " + shlex.quote(line) + " >> patch.patch")
            handlers["bash"].process("""$ patch --quiet -p1 < patch.patch""")
            handlers["bash"].process("""$ rm patch.patch""")

        elif block.lang in handlers:
            cli_logger.debug_log(f"Handling {block.lang} snippet")
            handlers[block.lang].process(text, block.extra)

    for language, handler in handlers.items():
        with TemporaryDirectory(
            prefix=f"revng-docs-test-{language}-{os.path.basename(path)}-"
        ) as temporary_directory:
            handler.run(Path(temporary_directory))


@click.command(name="test-docs", help="Test mkdocs files")
@click.argument("files", metavar="FILE...", nargs=-1, required=True)
def test_docs(files: tuple[str, ...]) -> None:
    for path in files:
        handle_file(Path(path))


def setup(registry: CommandRegistry):
    registry.register((), test_docs)
