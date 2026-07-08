/*
 * This file is distributed under the MIT License. See LICENSE.md for details.
 */

import { Tarball } from "@obsidize/tar-browserify";
import { saveAs } from "file-saver";
import { basename } from "path";
import { RefObject } from "preact";
import { createPortal } from "preact/compat";
import { useEffect, useLayoutEffect, useRef, useState } from "preact/hooks";

import { sleep } from "../utils/db";
import { ActionItem, Metadata } from "../utils/types";
import { Icon } from "./Icons";

async function openPerfetto(name: string, trace: string): Promise<void> {
    const req_promise = fetch(trace);
    const handle = window.open("https://ui.perfetto.dev", "_blank");
    if (handle === null) {
        return;
    }
    let ponged = false;
    window.addEventListener("message", function listener(ev) {
        if (ev.data === "PONG") {
            ponged = true;
            window.removeEventListener("message", listener);
        }
    });
    for (;;) {
        handle.postMessage("PING", "*");
        await sleep(100);
        if (ponged) {
            break;
        }
    }
    const req = await req_promise;
    if (!req.ok) {
        return;
    }
    handle.postMessage(
        {
            perfetto: {
                buffer: await req.arrayBuffer(),
                title: `Trace of ${name}`,
                fileName: basename(trace),
            },
        },
        "*"
    );
}

async function createReproducer(name: string, meta: Metadata): Promise<Uint8Array | undefined> {
    const binReq = await fetch(`${name}/input`);
    if (!binReq.ok) {
        return undefined;
    }
    const tarball = new Tarball();
    tarball.addBinaryFile("input", new Uint8Array(await binReq.arrayBuffer()), { fileMode: 0o444 });

    const commandReq = await fetch(`${name}/test-harness.json`);
    const command: string[] = (await commandReq.json()).command;

    for (let i = 0; i < command.length; i++) {
        if (command[i] == "%INPUT%") {
            command[i] = "input";
        }
    }

    const script = `#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(realpath "$(dirname "\${BASH_SOURCE[0]}")")

${meta.reproducer_prelude || ""}

${command.join(" ")} "$@"
`;
    tarball.addTextFile("go.sh", script, { fileMode: 0o555 });

    return tarball.toUint8Array();
}

function downloadUrl(url: string, filename?: string) {
    const a = document.createElement("a");
    a.href = url;
    if (filename) a.download = filename;
    document.body.append(a);
    a.click();
    a.remove();
}

export function rowActions(row: any, meta: Metadata, inDetail: boolean): ActionItem[] {
    const name: string = row.name;
    const items: ActionItem[] = [
        { label: "Download binary", run: () => downloadUrl(`${name}/input`, row.input_name) },
        { label: "View log", run: () => window.open(`${name}/output.log`, "_blank") },
    ];
    for (const dl of meta.downloads || []) {
        items.push({ label: dl.label, run: () => window.open(`${name}/${dl.name}`, "_blank") });
    }
    if (row.has_trace) {
        const trace = `${name}/trace.json.gz`;
        items.push({ label: "Download trace", run: () => downloadUrl(trace) });
        items.push({ label: "Open in Perfetto", run: () => openPerfetto(name, trace) });
    }
    items.push({
        label: "Reproduce",
        run: async () => {
            const tar = await createReproducer(name, meta);
            if (tar !== undefined) {
                saveAs(
                    new Blob([tar as any], { type: "application/tar" }),
                    `${basename(name)}-reproducer.tar`
                );
            }
        },
    });
    items.push({ label: "All files", run: () => window.open(`${name}/`, "_blank") });
    if (!inDetail) {
        items.push({ label: "-", run: () => {} });
        items.push({
            label: "Open details",
            strong: true,
            run: () => {
                window.location.href = `binary.html#${name}`;
            },
        });
    }
    return items;
}

export function ActionMenu({
    items,
    onRun,
    popup,
    popupRef,
}: {
    items: ActionItem[];
    onRun?: () => void;
    popup?: boolean;
    popupRef?: RefObject<HTMLDivElement>;
}) {
    return (
        <div ref={popupRef} class={popup ? "action-menu popup" : "action-menu"}>
            {items.map((item) =>
                item.label === "-" ? (
                    <div class="sep" />
                ) : (
                    <button
                        class={`action-item${item.strong ? " strong" : ""}`}
                        onClick={() => {
                            onRun?.();
                            item.run();
                        }}
                    >
                        {item.label}
                    </button>
                )
            )}
        </div>
    );
}

// A floating popup portalled to <body> so it escapes table overflow/stacking
// regardless of where the trigger button lives (including DataTables cells).
function ActionMenuPopup({
    anchor,
    items,
    onClose,
}: {
    anchor: RefObject<HTMLElement>;
    items: ActionItem[];
    onClose: () => void;
}) {
    const menuRef = useRef<HTMLDivElement>(null);

    // Position below the anchor, right-aligned, before paint.
    useLayoutEffect(() => {
        const r = anchor.current!.getBoundingClientRect();
        const menu = menuRef.current!;
        menu.style.top = `${r.bottom + 5}px`;
        menu.style.left = "auto";
        menu.style.right = `${window.innerWidth - r.right}px`;
    }, [anchor]);

    // Close on outside click / scroll / resize. Clicks on the anchor itself are
    // ignored so its own handler can toggle the menu.
    useEffect(() => {
        function onDocClick(ev: MouseEvent) {
            const target = ev.target as Node;
            if (!menuRef.current!.contains(target) && !anchor.current!.contains(target)) {
                onClose();
            }
        }
        document.addEventListener("click", onDocClick, true);
        document.addEventListener("scroll", onClose, true);
        window.addEventListener("resize", onClose);
        return () => {
            document.removeEventListener("click", onDocClick, true);
            document.removeEventListener("scroll", onClose, true);
            window.removeEventListener("resize", onClose);
        };
    }, [anchor, onClose]);

    return createPortal(
        <ActionMenu items={items} onRun={onClose} popup popupRef={menuRef} />,
        document.body
    );
}

export function ActionsButton({
    row,
    meta,
    inDetail,
}: {
    row: any;
    meta: Metadata;
    inDetail: boolean;
}) {
    const btn = useRef<HTMLButtonElement>(null);
    const [open, setOpen] = useState(false);
    return (
        <div style={{ display: "flex", justifyContent: "flex-end" }}>
            <button
                ref={btn}
                class="actions-btn"
                title="Actions"
                onClick={(ev) => {
                    ev.stopPropagation();
                    setOpen((o) => !o);
                }}
            >
                <Icon name="dots" />
            </button>
            {open && (
                <ActionMenuPopup
                    anchor={btn}
                    items={rowActions(row, meta, inDetail)}
                    onClose={() => setOpen(false)}
                />
            )}
        </div>
    );
}
