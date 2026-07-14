/*
 * This file is distributed under the MIT License. See LICENSE.md for details.
 */

import "./main.scss";

import { Database } from "@sqlite.org/sqlite-wasm";
import { render } from "preact";
import * as yaml from "yaml";

import { loadDBFromURL } from "./utils/db";
import { PAGES, PageConfig } from "./utils/pages";
import { Metadata } from "./utils/types";
import { BinaryDetail } from "./components/BinaryDetail";
import { Chrome } from "./components/Chrome";
import { Detail } from "./components/Detail";
import { ListPage } from "./components/ListPage";
import { Overview } from "./components/Overview";

function Page({ db, meta, cfg }: { db: Database; meta: Metadata; cfg: PageConfig }) {
    let body;
    if (cfg.page === "overview") {
        body = <Overview db={db} meta={meta} />;
    } else if (cfg.category) {
        body = <Detail db={db} meta={meta} cfg={cfg} />;
    } else {
        body = <ListPage db={db} meta={meta} cfg={cfg} />;
    }
    return (
        <Chrome meta={meta} cfg={cfg}>
            {body}
        </Chrome>
    );
}

function injectFont() {
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href =
        "https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap";
    document.head.append(link);
}

async function main() {
    injectFont();

    const db = await loadDBFromURL("main.db");
    window.db = db;

    const metaReq = await fetch("meta.yml");
    const meta: Metadata = yaml.parse(await metaReq.text());

    // Binary detail is a standalone page
    const detailRoot = document.getElementById("binary-detail");
    if (detailRoot !== null) {
        render(<BinaryDetail db={db} meta={meta} />, detailRoot);
        return;
    }

    const pageRoot = document.getElementById("page");
    if (pageRoot === null) {
        return;
    }
    const pageKey = pageRoot.getAttribute("data-page") || "overview";
    const cfg = PAGES[pageKey] || PAGES["overview"];
    render(<Page db={db} meta={meta} cfg={cfg} />, pageRoot);
}

main().then(() => {});
