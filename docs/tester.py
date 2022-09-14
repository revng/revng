#!/usr/bin/env python3

import json
import os
import marko
import re
import sys
import argparse
import subprocess

from tempfile import NamedTemporaryFile
from collections import defaultdict


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

def log(message):
    sys.stderr.write(message + "\n")


def only(list):
    assert len(list) == 1
    return list[0]


def run(arguments):
    log(str(arguments))
    return subprocess.run(arguments, check=True)


def main():
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument("files",
                        metavar="FILE",
                        nargs="+",
                        help="files to test.")
    args = parser.parse_args()

    handlers = {
        "python": PythonDoctest(),
        "bash": BashDoctest(),
        "typescript": TypeScriptDoctest()
    }

    scripts = defaultdict(str)

    for path in args.files:
        with open(path, "r") as input_file:
            document = marko.parse(input_file.read())

            skip_next = False
            for block in document.children:
                if type(block) is marko.block.HTMLBlock:
                    match = re.match(r"<!-- (.*?)( .*)? -->", block.children.strip())
                    if match:
                        groups = match.groups()
                        command = groups[0]
                        arguments = groups[1]
                        if command == "NO-TEST":
                            skip_next = True

                if type(block) is not marko.block.FencedCode:
                    continue

                rawtext = only(block.children)
                assert type(rawtext) is marko.inline.RawText
                text = rawtext.children
                assert type(text) is str

                match = re.match("title=(.*)", block.extra)
                if match:
                    append_to = match.groups()[0]
                    # WIP: delete file
                    scripts["bash"] += handlers["bash"].process(f"""$ cat <<'EOF' >> {append_to}
{text}EOF""")
                    append_to = None
                    continue

                if block.lang == "diff":
                    scripts["bash"] += handlers["bash"].process(f"""$ cat <<'EOF' > patch.patch
{text}EOF
$ patch --quiet -p1 < patch.patch
""")
                    continue

                if block.lang not in handlers:
                    continue

                if skip_next:
                    skip_next = False
                    continue

                scripts[block.lang] += handlers[block.lang].process(text)

    for language, script in scripts.items():
        handler = handlers[language]

        with NamedTemporaryFile("w",
                                suffix=handler.extension,
                                dir=os.getcwd()) as temporary_file:
            log(script)
            temporary_file.write(script)
            temporary_file.flush()
            run(handler.command(os.path.basename(temporary_file.name)))


if __name__ == "__main__":
    sys.exit(main())

