/*
 * This file is distributed under the MIT License. See LICENSE.md for details.
 */

import { Database } from "@sqlite.org/sqlite-wasm";
import { filesize } from "filesize";
import { useState } from "preact/hooks";

import {
    distinctSignatures,
    stageEntries,
    categoryFilters,
    statusCount,
    statusStats,
} from "../utils/db";
import { PageConfig } from "../utils/pages";
import { Metadata } from "../utils/types";
import { DataTableView } from "./DataTableView";
import { Flamegraphs } from "./Flamegraphs";
import { ComponentBars, FilterSelect } from "./ComponentBars";
import { SummaryCard } from "./SummaryCard";

function summaryNote(cfg: PageConfig, meta: Metadata, nSig: number): string {
    const conf = meta.configurations?.[0];
    if (cfg.status === "CRASHED") {
        return `across ${nSig} crash signatures`;
    } else if (cfg.status === "TIMED_OUT" && conf?.timeout) {
        return `${conf.timeout} s wall-clock limit - ${nSig} stacktraces`;
    } else if (cfg.status === "OOM" && conf?.memory_limit) {
        return `${filesize(conf.memory_limit, { base: 2 })} memory limit - ${nSig} stacktraces`;
    }
    return `across ${nSig} stacktraces`;
}

export function Detail({ db, meta, cfg }: { db: Database; meta: Metadata; cfg: PageConfig }) {
    const stats = statusStats(db);
    const count = statusCount(stats, cfg.status!);
    const nSig = distinctSignatures(db, cfg.status!);
    const note = summaryNote(cfg, meta, nSig);

    const filters = categoryFilters(meta, cfg.category!);
    const [filter, setFilter] = useState("all");
    const entries = stageEntries(db, cfg.category!, filter);

    return (
        <>
            <div class="summary-grid">
                <SummaryCard cfg={cfg} count={count} total={stats.total} note={note} />
                <div class={`card ${cfg.accent}`}>
                    <div class="comp-head" style={{ marginBottom: "14px" }}>
                        <span class="card-label">By component</span>
                        {filters.length > 1 ? (
                            <FilterSelect filters={filters} value={filter} onChange={setFilter} />
                        ) : null}
                    </div>
                    <ComponentBars entries={entries} limit={8} twoCol={true} />
                </div>
            </div>

            <div class="fg-card">
                <div class="fg-title">Flamegraphs</div>
                <div class="fg-hint">
                    {`Aggregated stack traces - ${cfg.fgEnd} location at the top`}
                </div>
                <Flamegraphs prefix={cfg.fgPrefix!} filter={filter} end={cfg.fgEnd!} />
            </div>

            <DataTableView db={db} meta={meta} cfg={cfg} />
        </>
    );
}
