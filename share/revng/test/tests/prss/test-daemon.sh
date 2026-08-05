#!/usr/bin/env bash

#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

set -euo pipefail
INPUT_BINARY="$1"

SCRIPT_DIR=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common"

start_postgres
start_relay_server
start_rss_server

PROJECT_ID=$(python -c 'from uuid import uuid4; print(str(uuid4()))')

# Set up a script which takes care of dumping notifications from websocket
mkdir "$WORKDIR/messages"
"$SCRIPT_DIR/websocket_client.py" \
    "ws://127.0.0.1:$RELAY_NOTIFICATIONS_PORT/notifications?project_id=${PROJECT_ID}" \
    "$WORKDIR/messages" &
PIDS_TO_KILL+=($!)

# Start the revng daemon
DAEMON_PORT=$(available_port)
revng project \
    --storage-provider "rss://127.0.0.1:$RSS_SERVER_PORT/?proto=http" \
    daemon \
    --bind "127.0.0.1:$DAEMON_PORT" \
    &> "$WORKDIR/daemon.log" &
PIDS_TO_KILL+=($!)
wait_for_status "$DAEMON_PORT"

# Run the script that uses the daemon HTTP API to request it to run the initial
# auto analysis
"$SCRIPT_DIR/../scripting.py" \
    --project-id "$PROJECT_ID" \
    --daemon-url "http://127.0.0.1:$DAEMON_PORT" \
    --binary "$INPUT_BINARY"

# Check at least one invalidation was received and that all of them have the
# same format
for INDEX in {0..20}; do
    MESSAGE_FILE="$WORKDIR/messages/message$INDEX"
    if [[ ! -e "$MESSAGE_FILE" ]]; then
        break
    fi

    # Check that the message is JSON
    jq . "$MESSAGE_FILE" > /dev/null
    # Assert that some data is correct
    [[ $(jq -r .type "$MESSAGE_FILE") = "invalidation" ]]
    [[ $(jq -r .epoch "$MESSAGE_FILE") -eq $((INDEX+1)) ]]
    [[ $(jq -r .invalidated "$MESSAGE_FILE") != "null" ]]
done
