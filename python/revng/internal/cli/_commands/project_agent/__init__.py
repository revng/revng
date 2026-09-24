#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import json
import os
import re
import shutil
import signal
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Literal, Protocol

import click
import jwt
from jinja2 import Template

from revng.internal.cli.common import CommandRegistry, cli_logger
from revng.internal.support import cache_directory, check_unix_socket
from revng.pypeline.cli.context import ClickContext, pass_context
from revng.pypeline.storage.storage_provider import storage_provider_factory_factory
from revng.support import get_root

# Re-run auto-detection when the cached check is older than this
CACHE_MAX_AGE = 24 * 60 * 60  # 24 hours
# Plan values that do not count as a subscription
INACTIVE_PLANS = (None, "", "free")


def _run(command: list[str], ignore_status: bool = False) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except (OSError, subprocess.SubprocessError):
        return None

    if result.returncode != 0 and not ignore_status:
        return None

    return result.stdout


def _parse_json(data: str) -> Any:
    try:
        return json.loads(data)
    except ValueError:
        return None


def _walk_dicts(value: Any, *keys: str) -> Any:
    """Walk nested dictionaries, returning None on any missing or non-dict step."""
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)

    return value


def _read_jwt(token: Any) -> dict | None:
    try:
        return jwt.decode(token, options={"verify_signature": False})
    except jwt.InvalidTokenError:
        return None


def _get_chatgpt_plan(claims: dict) -> Any:
    """The plan advertised by an OpenAI token, None for any other issuer."""
    return _walk_dicts(claims, "https://api.openai.com/auth", "chatgpt_plan_type")


class Agent(Protocol):
    name: str

    def detect(self) -> int | Literal[True] | None: ...
    def get_command(self, prompt: str) -> list[str]: ...


class Claude:
    name = "claude"

    @staticmethod
    def detect() -> Literal[True] | None:
        """`claude auth status` reports the plan itself and exposes no token."""
        output = _run(["claude", "auth", "status", "--json"])
        if output is None:
            return None

        status = _parse_json(output)
        # An API key is not a subscription
        if not status.get("loggedIn") or status.get("authMethod") != "claude.ai":
            return None

        if status.get("subscriptionType") in INACTIVE_PLANS:
            return None

        return True

    @staticmethod
    def get_command(prompt: str) -> list[str]:
        return ["claude", prompt]


class Codex:
    name = "codex"

    @staticmethod
    def detect() -> int | None:
        """`codex doctor --json` reports the login mode and the auth file, untruncated."""
        # Unrelated failing checks (e.g. network) must not hide the auth report
        output = _run(["codex", "doctor", "--json"], ignore_status=True)
        if output is None:
            return None

        report = _parse_json(output)
        details = _walk_dicts(report, "checks", "auth.credentials", "details")
        # The alternatives are an API key login or no login at all
        if _walk_dicts(details, "stored auth mode") != "chatgpt":
            return None

        if (auth_file := details.get("auth file")) is None:
            return None

        auth_file_data = _parse_json(Path(auth_file).read_text())
        claims = _read_jwt(_walk_dicts(auth_file_data, "tokens", "id_token"))
        if claims is None or _get_chatgpt_plan(claims) in INACTIVE_PLANS:
            return None

        return claims["exp"]

    @staticmethod
    def get_command(prompt: str) -> list[str]:
        return ["codex", "--dangerously-bypass-approvals-and-sandbox", prompt]


class Opencode:
    name = "opencode"

    @staticmethod
    def detect() -> int | None:
        """opencode has no notion of a plan, so fall back to its stored credentials."""
        if (output := _run(["opencode", "debug", "paths"])) is None:
            return None

        data_dir_re = re.compile(r"^data\s+(\S+)$")
        for line in output.splitlines():
            if (match := data_dir_re.match(line)) is not None:
                break
        else:
            return None

        credentials = _parse_json((Path(match.group(1)) / "auth.json").read_text())
        if not isinstance(credentials, dict):
            return None

        for credential in credentials.values():
            if not isinstance(credential, dict) or credential.get("type") != "oauth":
                continue

            claims = _read_jwt(credential.get("access"))
            # Only OpenAI advertises a plan, take any other provider at face value
            if claims is None or _get_chatgpt_plan(claims) in INACTIVE_PLANS:
                continue

            return claims["exp"]

        return None

    @staticmethod
    def get_command(prompt: str) -> list[str]:
        return ["opencode", "--prompt", prompt]


AGENTS: dict[str, Agent] = {a.name: a for a in (Claude, Codex, Opencode)}


def _read_agent_from_cache(cache_file: Path) -> str | None:
    cache = {}
    if cache_file.is_file():
        cache = _parse_json(cache_file.read_text())

    if cache == {} or cache["agent"] not in AGENTS:
        return None

    now = time.time()
    if now - cache["check-time"] > CACHE_MAX_AGE or cache["expiry"] <= now:
        return None

    return cache["agent"]


def _detect_agent() -> Agent:
    cache_file = cache_directory() / "agent-subscription.json"
    cached_agent_name = _read_agent_from_cache(cache_file)
    if cached_agent_name is not None:
        cli_logger.debug_log(f'Using cached agent: "{cached_agent_name}"')
        return AGENTS[cached_agent_name]

    for agent in AGENTS.values():
        subscription = agent.detect()
        if subscription is not None:
            payload = {
                "agent": agent.name,
                "expiry": None if subscription is True else subscription,
                "check-time": int(time.time()),
            }
            cache_file.write_text(json.dumps(payload))
            return agent

    raise click.ClickException(
        f"None of {', '.join(AGENTS)} has an active subscription. "
        "Use --agent to pick one anyway."
    )


@contextmanager
def _daemon(ctx: ClickContext, socket_path: Path) -> Generator[None]:
    """Start a daemon unless one already answers on `socket_path`, and stop it afterwards."""
    if check_unix_socket(socket_path):
        cli_logger.debug_log(f"Reusing the daemon on {socket_path}")
        yield
        return

    process = subprocess.Popen(["revng", "-C", str(ctx.obj.base_directory), "project", "daemon"])
    try:
        while not check_unix_socket(socket_path):
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
    help="Use the specified agent, skipping subscription auto-detection.",
)
@click.option("--prompt-append", help="Append the specified string to the prompt")
@pass_context
def project_agent(ctx: ClickContext, agent: str | None, prompt_append: str | None):
    """Run a coding agent against the project's daemon."""

    factory = storage_provider_factory_factory(ctx.obj.storage_provider_url)
    model_path = factory.model_path(ctx.obj.base_directory)
    if model_path is None:
        raise click.UsageError("The storage provider does not have a model path, bailing.")

    socket_path = model_path.parent / "revng.sock"
    if agent is not None:
        if shutil.which(agent) is None:
            raise click.ClickException(f"{agent} is not installed")
        agent_class = AGENTS[agent]
    else:
        agent_class = _detect_agent()

    with open(Path(__file__).parent / "prompt.tpl") as f:
        template = Template(f.read())

    doc_root = str(get_root() / "share/doc/revng")
    prompt = template.render({"doc_root": doc_root})
    if prompt_append is not None:
        prompt += "\n"
        prompt += prompt_append

    command = agent_class.get_command(prompt)
    env = {**os.environ, "REVNG_STORAGE_PROVIDER": f"daemon://!unix{socket_path.resolve()!s}"}
    with _daemon(ctx, socket_path):
        return subprocess.run(command, cwd=model_path.parent, env=env, check=False).returncode


def setup(registry: CommandRegistry):
    registry.register(("project",), project_agent)
