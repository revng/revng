#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import hashlib
import json
import os
import shutil
import signal
import string
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, NotRequired, TypedDict

import click
import yaml
from jinja2 import Template

from revng.internal.cli.common import CommandRegistry, cli_logger
from revng.internal.support import cache_directory, is_unix_socket_available
from revng.pypeline.cli.context import ClickContext, pass_context
from revng.pypeline.storage.storage_provider import storage_provider_factory_factory
from revng.support import get_root


class AgentEntry(TypedDict):
    # Command to run the agent, if a string contains a $var placeholder it will
    # be replaced
    command: list[str]
    # Additional arguments if a user prompt has been specified
    user_prompt_args: list[str]
    # Environment variables to use to run the agent. These also have variable
    # placeholders replaced before passing them to the process.
    environment: NotRequired[dict[str, str]]


def _load_agents() -> dict[str, AgentEntry]:
    with open(get_root() / "share/revng/agents.yml") as f:
        return yaml.safe_load(f)


AGENTS = _load_agents()


def _detect_or_ask_agent():
    cache_file = cache_directory() / "agent"
    if cache_file.is_file():
        agent_name = cache_file.read_text()
        if agent_name in AGENTS:
            return AGENTS[agent_name]

    def save_agent(agent_name):
        cache_file.write_text(agent_name)

    agents_present = []
    for agent, agent_config in AGENTS.items():
        if shutil.which(agent_config["command"][0]) is not None:
            agents_present.append(agent)

    if len(agents_present) == 0:
        raise click.ClickException(f"None of {', '.join(AGENTS)} are present on this system.")
    elif len(agents_present) == 1:
        save_agent(agents_present[0])
        return AGENTS[agents_present[0]]
    else:
        print("Multiple agents detected, pick one from the following list:")
        for index, agent_name in enumerate(agents_present):
            print(f"{index + 1}. {agent_name}")
        while True:
            choice = input("> ")
            try:
                index_value = int(choice)
            except ValueError:
                index_value = None

            if index_value is None or index_value < 1 or index_value > len(agents_present):
                print("Invalid choice")
                continue

            agent_name = agents_present[index_value - 1]
            save_agent(agent_name)
            return AGENTS[agent_name]


@contextmanager
def _daemon(ctx: ClickContext, socket_path: Path) -> Generator[None]:
    """Start a daemon unless one already answers on `socket_path`, and stop it afterwards."""
    if is_unix_socket_available(socket_path):
        cli_logger.debug_log(f"Reusing the daemon on {socket_path}")
        yield
        return

    process = subprocess.Popen(
        [
            "revng",
            "-C",
            str(ctx.obj.base_directory),
            "project",
            "--storage-provider",
            ctx.obj.storage_provider_url,
            "daemon",
        ]
    )
    try:
        while not is_unix_socket_available(socket_path):
            if (returncode := process.poll()) is not None:
                raise click.ClickException(f"`revng project daemon` exited with {returncode}")
            time.sleep(0.1)
        yield
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(10.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


@click.command(name="agent")
@click.option(
    "--agent",
    type=click.Choice(AGENTS),
    help="Use the specified agent, skipping auto-detection.",
)
@click.option("--system-prompt-append", help="Append the specified string to the system prompt.")
@click.option("--user-prompt", help="Start the agent with the specified user prompt.")
@pass_context
def project_agent(
    ctx: ClickContext,
    agent: str | None,
    system_prompt_append: str | None,
    user_prompt: str | None,
):
    """Run a coding agent against the project's daemon."""

    if agent is not None:
        agent_config = AGENTS[agent]
        if shutil.which(agent_config["command"][0]) is None:
            raise click.ClickException(f"{agent} is not installed")
    else:
        agent_config = _detect_or_ask_agent()

    factory = storage_provider_factory_factory(ctx.obj.storage_provider_url)
    model_path = factory.model_path(ctx.obj.base_directory)
    if model_path is None:
        raise click.UsageError("The storage provider does not have a model path, bailing.")

    socket_path = model_path.parent / "revng.sock"

    with open(Path(__file__).parent / "prompt.tpl") as f:
        template = Template(f.read())

    doc_root = str(get_root() / "share/doc/revng")
    instructions = template.render(
        {"doc_root": doc_root, "system_prompt_append": system_prompt_append}
    )

    # Write the instructions to a file, needed for some agents
    instructions_hash = hashlib.sha256(instructions.encode("utf-8")).hexdigest()[:8]
    instructions_file = cache_directory() / "prompts" / f"prompt_{instructions_hash}.md"
    instructions_file.parent.mkdir(parents=True, exist_ok=True)
    instructions_file.write_text(instructions)

    variables = {
        "instructions": instructions,
        # A JSON string is also a valid TOML basic string
        "instructions_toml": json.dumps(instructions),
        "instructions_file": str(instructions_file),
        "user_prompt": user_prompt,
    }

    def expand(value: str) -> str:
        return string.Template(value).substitute(variables)

    command = list(agent_config["command"])
    if user_prompt is not None:
        command.extend(agent_config["user_prompt_args"])
    command = [expand(a) for a in command]

    env = {
        **os.environ,
        **{k: expand(v) for k, v in agent_config.get("environment", {}).items()},
        "REVNG_STORAGE_PROVIDER": f"daemon://!unix{socket_path.resolve()!s}",
    }
    with _daemon(ctx, socket_path):
        return subprocess.run(command, cwd=model_path.parent, env=env, check=False).returncode


@click.command(name="detect-agents")
def detect_agents():
    for agent_name, agent_config in AGENTS.items():
        if shutil.which(agent_config["command"][0]) is not None:
            sys.stdout.write(agent_name + "\n")


def setup(registry: CommandRegistry):
    registry.register(("project",), project_agent)
    registry.register(("internal",), detect_agents)
