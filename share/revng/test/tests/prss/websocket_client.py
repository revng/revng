#!/usr/bin/env python3

#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import signal
from pathlib import Path

import click
from websockets.sync.client import connect


@click.command()
@click.argument("ws_url")
@click.argument("output_dir", type=click.Path(file_okay=False, writable=True, path_type=Path))
def main(ws_url: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    with connect(ws_url) as ws:
        signal.signal(signal.SIGINT, lambda *args: ws.close())
        for index, message in enumerate(ws):
            output_path = output_dir / f"message{index}"
            if isinstance(message, bytes):
                output_path.write_bytes(message)
            else:
                output_path.write_text(message)


if __name__ == "__main__":
    main()
