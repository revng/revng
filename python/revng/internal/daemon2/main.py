#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import os
import sys
import logging
import importlib
import importlib.util
from pathlib import Path

import json
import yaml
import click
import uvicorn
import jsonschema

import revng
from revng.pypeline.pipeline_parser import load_pipeline_yaml_file

from .app import app

logger = logging.getLogger("pypeline")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stderr))

with open(Path(__file__).parent / "config_schema.json") as f:
    CONFIGS_SCHEMA = json.load(f)

PIPELINE_PATH = Path(__file__).parent.parent / "pypeline.yml"

CONFIG_PATH = Path(__file__).parent.parent / "daemon.yml"

@click.command()
@click.option("--host", required=False, help="The interface to bind to.")
@click.option("--port", type=int, required=False, help="The port to bind to.")
@click.option("--reload", is_flag=True, help="Enable auto-reload on file changes.")
@click.option("--workers", type=int, required=False, help="Number of worker processes.")
# TODO: expose all other uvicorn parameters from https://github.com/Kludex/uvicorn/blob/master/uvicorn/main.py
@click.option(
    "--pipeline",
    "pipeline_path",
    metavar="PIPELINE_PATH",
    type=click.Path(exists=True, readable=True),
    default=PIPELINE_PATH,
    show_default=True,
    help="The pipeline to load",
)
@click.option(
    "--config",
    "config_path",
    metavar="CONFIG_PATH",
    type=click.Path(exists=True, readable=True),
    default=CONFIG_PATH,
    show_default=True,
    help="The daemon config to load",
)
@click.pass_context
def run_server(ctx, host, port, reload, workers, pipeline_path, config_path):
    """Runs the async revng-daemon with uvicorn."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Check that the configs are reasonable,
    validator = jsonschema.Draft7Validator(CONFIGS_SCHEMA)
    validator.validate(config)

    # Load the pipebox and pipeline configuration
    pipeline = load_pipeline_yaml_file(pipeline_path)
    # Store in the app state so all routes can access them
    app.state.pipeline = pipeline
    app.state.pipebox = ctx.obj["pipebox"]

    with open(pipeline_path, "r", encoding="utf-8") as f:
        app.state.pipeline_raw = f.read()

    app.state.storage_provider = config["storage_provider"]
    app.state.version = revng.__version__

    # The lock class follows the same syntax as our click's LazyGroup
    # <module>:<type_name>
    lock_module_name, lock_ty_name = config["lock"]["class"].rsplit(":", 1)
    lock_module = importlib.import_module(lock_module_name)
    lock_ty = getattr(lock_module, lock_ty_name)
    app.state.lock = lock_ty(**config["lock"].get("args", {}))

    # Configure uvicorn logging
    log_config = uvicorn.config.LOGGING_CONFIG

    # Setup formatting
    log_format = config.get("logging", {}).get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log_config["formatters"]["access"]["fmt"] = log_format
    log_config["formatters"]["default"]["fmt"] = log_format

    # Apply log level to handlers if specified
    log_level = config.get("logging", {}).get("level", None)
    if log_level:
        for uvlogger in log_config["loggers"].values():
            uvlogger["level"] = log_level.upper()

    # Apply config file settings if user didn't override them through cli args
    if "server" in config:
        if host is None:
            host = config["server"].get("host", "127.0.0.1")
        if port is None:
            port = config["server"].get("port", "5000")
        if workers is None:
            workers = config["server"].get("workers", 1)

    if workers is None:
        workers = 1

    # Start the uvicorn server
    uvicorn.run(
        "revng.internal.daemon2.app:app",
        host=host,
        port=int(port),
        reload=reload,
        workers=int(workers) if not reload else 1,  # reload doesn't work with multiple workers
        log_config=log_config,
        access_log=True,
    )
