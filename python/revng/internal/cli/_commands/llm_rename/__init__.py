#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import csv
import os
import re
from io import StringIO
from pathlib import Path
from typing import cast

import click
import requests
import yaml
from jinja2 import Template

from revng.internal.cli.common import CommandRegistry, cli_logger
from revng.internal.cli.support import file_wrapper
from revng.ptml import parse
from revng.ptml.actions import ACTION_MAPPING, model_path_for_rename
from revng.support.location import Location

LOCATIONS_WITH_RENAME = [key for key, value in ACTION_MAPPING.items() if value.rename is not None]
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class LLMRenameException(Exception):
    def __init__(self, text: str):
        super().__init__(text)
        self.text = text


class LLMRenameRetriableException(LLMRenameException):
    pass


def _parse_input(path: str | None) -> tuple[str, dict[str, str]]:
    with file_wrapper(path, "rb") as input_file:
        document = parse(input_file)

    location_mapping = {}
    for ra in document.metadata.walk():
        if "data-action-context-location" not in ra.attributes:
            continue
        location = ra.attributes["data-action-context-location"]
        if not any(location.startswith(f"/{x}/") for x in LOCATIONS_WITH_RENAME):
            continue
        location_text = document.extractor.get_text(ra.text_range)
        location_mapping[location_text] = location

    return document.text, location_mapping


def _parse_response(text: str) -> dict[str, str]:
    if not (
        (text.startswith("```csv\n") or text.startswith("```\n"))
        and text.endswith("```")
        and text.count("```") == 2
    ):
        raise LLMRenameRetriableException(f"Response is not a csv code block:\n{text}")

    text_cleaned = "\n".join(text.split("\n")[1:-1])
    try:
        text_parsed = list(csv.DictReader(StringIO(text_cleaned)))
    except csv.Error as e:
        raise LLMRenameRetriableException(f"Malformed response from model:\n{text}") from e

    result = {}
    for row in text_parsed:
        replacement = row["new"].strip()
        if IDENTIFIER_RE.match(replacement) is None:
            cli_logger.log(f"Dropping incompatible identifier '{replacement}'")
            continue
        result[row["old"]] = replacement
    return result


def _send_and_parse_response(text: str) -> dict[str, str]:
    req = requests.post(
        "https://api.openai.com/v1/responses",
        json={"model": "gpt-4o-mini", "input": text},
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
    )
    if not req.ok:
        raise LLMRenameException(
            f"Request to OpenAI returned response {req.status_code}\n{req.text}"
        )

    data = req.json()
    response = None
    for entry in data["output"]:
        if entry["type"] == "message" and entry["role"] == "assistant":
            response = entry
            break

    if response is None:
        raise LLMRenameRetriableException("No valid response returned from request")

    return _parse_response(response["content"][0]["text"])


def _produce_output(output: str | None, location_mapping: dict[str, str], data: dict[str, str]):
    result = []
    for old, new in data.items():
        if (location := location_mapping.get(old)) is None:
            continue
        if (location_parsed := Location.parse(location)) is None:
            continue
        if (model_path := model_path_for_rename(location_parsed)) is None:
            continue
        result.append({"Path": model_path, "Add": new})

    with file_wrapper(output, "w") as output_file:
        yaml.safe_dump({"Changes": result}, output_file)


# This command will try and "rename" a piece of PTML, by feeding a suitable
# prompt to an LLM and parse the response.
#
# Input parameters:
# * The input file: a PTML file
# * The OPENAI_API_KEY environment variable: a valid OpenAI key
#
# Output:
# If the exit code is 0 then the command returned a valid mapping to rename
# the variables contained in the PTML document, with the format:
# ```yaml
# Changes:
# - Path: /Tuple/Tree/Path/Of/The/Model
#   Add: new_value_to_use   # noqa: E800
# ```
# This format is the same as the output of `revng model diff` with the omission
# of the `Remove` field.
#
# If the exit code is 2 then the stderr of this program can be used as the
# error message for downstream users.
@click.command(name="llm-rename", help="Rename the provided PTML with LLM")
@click.argument("input_", metavar="[INPUT]", required=False)
@click.argument("output", metavar="[OUTPUT]", required=False)
@click.option("--no-send", is_flag=True, help="Do not send the prompt, just print it")
def llm_rename(input_: str | None, output: str | None, no_send: bool) -> int:
    if not no_send and "OPENAI_API_KEY" not in os.environ:
        cli_logger.log("OPENAI_API_KEY not found in environment variables, aborting")
        return 2

    text, location_mapping = _parse_input(input_)
    if len(location_mapping) == 0:
        return 0

    with open(Path(__file__).parent / "prompt.tpl") as f:
        template = Template(f.read())

    csv_text = "".join(f"{k},\n" for k in location_mapping.keys())
    rendered_text = template.render({"code": text, "csv": csv_text})
    if no_send:
        print(rendered_text)
        return 0

    response_data = None
    last_retriable_error = None
    for _ in range(0, 3):
        try:
            response_data = _send_and_parse_response(rendered_text)
        except LLMRenameRetriableException as e:
            last_retriable_error = e
        except LLMRenameException as e:
            cli_logger.log(e.text)
            return 2

    if response_data is None:
        cli_logger.log(cast(LLMRenameException, last_retriable_error).text)
        return 2

    _produce_output(output, location_mapping, response_data)
    return 0


def setup(registry: CommandRegistry):
    registry.register((), llm_rename)
