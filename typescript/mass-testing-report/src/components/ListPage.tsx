/*
 * This file is distributed under the MIT License. See LICENSE.md for details.
 */

import { Database } from "@sqlite.org/sqlite-wasm";

import { statusCount, statusStats } from "../utils/db";
import { PageConfig } from "../utils/pages";
import { Metadata } from "../utils/types";
import { DataTableView } from "./DataTableView";
import { SummaryCard } from "./SummaryCard";

export function ListPage({ db, meta, cfg }: { db: Database; meta: Metadata; cfg: PageConfig }) {
    const stats = statusStats(db);
    return (
        <>
            {cfg.page === "raw_data" ? (
                <div class="section-head" style={{ marginTop: "0" }}>
                    <span class="title">Raw data</span>
                    <a href="main.db" download="main.db">
                        Download raw DB
                    </a>
                </div>
            ) : cfg.status ? (
                <div class="summary-grid" style={{ gridTemplateColumns: "1fr" }}>
                    <SummaryCard
                        cfg={cfg}
                        count={statusCount(stats, cfg.status)}
                        total={stats.total}
                    />
                </div>
            ) : null}

            <DataTableView db={db} meta={meta} cfg={cfg} />
        </>
    );
}
