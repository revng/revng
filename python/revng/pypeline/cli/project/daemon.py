#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import os

import click
import uvicorn
import yaml

import revng
from revng.pypeline.daemon.app import make_starlette
from revng.pypeline.daemon.daemon import Daemon


@click.command()
@click.option(
    "--production",
    is_flag=True,
    help="Enable production settings.",
)
@click.pass_context
def run_daemon(ctx, production, **kwargs):
    """Start the HTTP daemon."""

    with open(ctx.obj["pipeline_path"], "r", encoding="utf-8") as f:
        pipeline_yaml = yaml.safe_load(f.read())

    # Configure uvicorn logging
    log_config = uvicorn.config.LOGGING_CONFIG

    # Setup formatting
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_config["formatters"]["access"]["fmt"] = log_format
    log_config["formatters"]["default"]["fmt"] = log_format

    for uvlogger in log_config["loggers"].values():
        uvlogger["level"] = "INFO"

    if not production:
        os.environ["STARLETTE_DEBUG"] = "1"
        os.environ["REVNG_ORIGINS"] = "*"
        kwargs.setdefault("host", "127.0.0.1")
    else:
        kwargs.setdefault("host", "0.0.0.0")

    daemon = Daemon(
        version=revng.__version__,
        pipeline_yaml=pipeline_yaml,
        pipeline=ctx.obj["pipeline"],
        debug=not production,
        storage_provider_url=ctx.obj["storage_provider"],
        cache_dir=ctx.obj["cache_dir"],
    )

    # Start the uvicorn server
    uvicorn.run(app=make_starlette(daemon), **kwargs)


# Inherit all params from the uvicorn cli
for param in uvicorn.main.params:
    # ignore the app param because we will pass it in `run_daemon`
    if param.name == "app":
        continue
    # Add help text that explains production changes
    if param.name == "host":
        param.default = None  # type: ignore [attr-defined]
        param.show_default = False  # type: ignore [attr-defined]
        param.help += (  # type: ignore [attr-defined]
            " Defaults to 0.0.0.0 in production and 127.0.0.1 otherwise."
        )
    run_daemon.params.append(param)
