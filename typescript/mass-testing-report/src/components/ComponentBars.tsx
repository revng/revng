/*
 * This file is distributed under the MIT License. See LICENSE.md for details.
 */

import { Database } from "@sqlite.org/sqlite-wasm";
import { useMemo, useState } from "preact/hooks";

import { stageEntries, categoryFilters } from "../utils/db";
import { percent } from "../utils/format";
import { Metadata, StageEntry } from "../utils/types";

export function ComponentBars({
    entries,
    limit,
    twoCol,
}: {
    entries: StageEntry[];
    limit: number;
    twoCol: boolean;
}) {
    const shown = entries.slice(0, limit);
    const total = entries.reduce((a, e) => a + e.count, 0) || 1;
    const max = Math.max(1, ...shown.map((e) => e.count));

    return (
        <div class={twoCol ? "bars two-col" : "bars"}>
            {shown.map((e) => (
                <div class="bar">
                    <div class="bar-top">
                        <span class="bar-name" title={e.name}>
                            {e.name}
                        </span>
                        <span class="bar-val">{`${e.count} - ${percent(e.count, total)}`}</span>
                    </div>
                    <div class="bar-track">
                        <div class="bar-fill" style={{ width: `${(e.count / max) * 100}%` }} />
                    </div>
                </div>
            ))}
        </div>
    );
}

export function FilterSelect({
    filters,
    value,
    onChange,
}: {
    filters: [string, string][];
    value: string;
    onChange: (filter: string) => void;
}) {
    return (
        <div class="pctl-row">
            <span class="lbl">Filter</span>
            <select
                class="pctl"
                value={value}
                onChange={(ev) => onChange((ev.target as HTMLSelectElement).value)}
            >
                {filters.map((f) => (
                    <option value={f[0]} selected={f[0] === value}>
                        {f[1]}
                    </option>
                ))}
            </select>
        </div>
    );
}

export function CrashComponentsCard({
    db,
    meta,
    label,
    category,
    accent,
    headCount,
    limit,
}: {
    db: Database;
    meta: Metadata;
    label: string;
    category: string;
    accent: string;
    headCount: number;
    limit: number;
}) {
    const filters = useMemo(() => categoryFilters(meta, category), [category]);
    const [filter, setFilter] = useState("all");
    const entries = stageEntries(db, category, filter);
    const initialCount = useMemo(() => stageEntries(db, category, "all").length, [category, "all"]);

    return (
        <div class={`card stage-card ${accent}`}>
            <div class="stage-card-head">
                <span class="dot" />
                <span class="stage-card-title">{label}</span>
                <span class="stage-card-count">{headCount}</span>
            </div>
            {filters.length > 1 ? (
                <FilterSelect filters={filters} value={filter} onChange={setFilter} />
            ) : null}
            <ComponentBars entries={entries} limit={limit} twoCol={false} />
            {initialCount > limit ? (
                <div class="stage-more">{`+ ${initialCount - limit} more components`}</div>
            ) : null}
        </div>
    );
}
