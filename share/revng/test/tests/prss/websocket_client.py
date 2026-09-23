#!/usr/bin/env python3

#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import signal
from pathlib import Path

import click
import httpx2
from httpx2.websockets import WebSocketDisconnect
from wsproto.events import BytesMessage, TextMessage


@click.command()
@click.argument("ws_url")
@click.argument("output_dir", type=click.Path(file_okay=False, writable=True, path_type=Path))
def main(ws_url: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    stopped = False

    def stop(*args):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, stop)
    with httpx2.websocket(ws_url) as ws:
        index = 0
        while not stopped:
            try:
                # The timeout lets the loop notice that SIGINT was delivered
                message = ws.receive(timeout=1.0)
            except TimeoutError:
                continue
            except WebSocketDisconnect:
                break

            output_path = output_dir / f"message{index}"
            if isinstance(message, BytesMessage):
                output_path.write_bytes(message.data)
            elif isinstance(message, TextMessage):
                output_path.write_text(message.data)
            else:
                continue
            index += 1


if __name__ == "__main__":
    main()
