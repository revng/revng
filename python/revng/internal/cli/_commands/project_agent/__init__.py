#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import os
import shutil
import signal
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import click
import yaml
from jinja2 import Template

from revng.internal.cli.common import CommandRegistry, cli_logger
from revng.internal.support import cache_directory, is_unix_socket_available
from revng.pypeline.cli.context import ClickContext, pass_context
from revng.pypeline.storage.storage_provider import storage_provider_factory_factory
from revng.support import get_root


def _load_agents() -> dict[str, list[str]]:
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
    for agent, agent_cmd in AGENTS.items():
        if shutil.which(agent_cmd[0]) is not None:
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
@click.option("--prompt-append", help="Append the specified string to the prompt.")
@pass_context
def project_agent(ctx: ClickContext, agent: str | None, prompt_append: str | None):
    """Run a coding agent against the project's daemon."""

    if agent is not None:
        if shutil.which(agent) is None:
            raise click.ClickException(f"{agent} is not installed")
        agent_cmd = AGENTS[agent]
    else:
        agent_cmd = _detect_or_ask_agent()

    factory = storage_provider_factory_factory(ctx.obj.storage_provider_url)
    model_path = factory.model_path(ctx.obj.base_directory)
    if model_path is None:
        raise click.UsageError("The storage provider does not have a model path, bailing.")

    socket_path = model_path.parent / "revng.sock"

    with open(Path(__file__).parent / "prompt.tpl") as f:
        template = Template(f.read())

    doc_root = str(get_root() / "share/doc/revng")
    prompt = template.render({"doc_root": doc_root})
    if prompt_append is not None:
        prompt += "\n"
        prompt += prompt_append

    command = (*agent_cmd, prompt)
    env = {**os.environ, "REVNG_STORAGE_PROVIDER": f"daemon://!unix{socket_path.resolve()!s}"}
    with _daemon(ctx, socket_path):
        return subprocess.run(command, cwd=model_path.parent, env=env, check=False).returncode


def setup(registry: CommandRegistry):
    registry.register(("project",), project_agent)
