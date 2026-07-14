/*
 * This file is distributed under the MIT License. See LICENSE.md for details.
 */

import { Database } from "@sqlite.org/sqlite-wasm";

import { sqlOptions, statusCount, statusStats } from "../utils/db";
import { formatClock, formatHM, formatTimestamp, percent } from "../utils/format";
import { Metadata } from "../utils/types";
import { Icon } from "./Icons";
import { CrashComponentsCard } from "./ComponentBars";

const KPI_DEFS: [string, string, string, string | null][] = [
    ["Failures", "FAILED", "a-fail", null],
    ["Crashes", "CRASHED", "a-crash", "crashes.html"],
    ["Timeouts", "TIMED_OUT", "a-timeout", "timeouts.html"],
    ["OOMs", "OOM", "a-oom", "ooms.html"],
];

const COMPOSITION: [string, string, string][] = [
    ["Successes", "OK", "var(--ok)"],
    ["Timeouts", "TIMED_OUT", "var(--timeout)"],
    ["Crashes", "CRASHED", "var(--crash)"],
    ["OOMs", "OOM", "var(--oom)"],
    ["Failures", "FAILED", "var(--fail)"],
];

const STAGE_DEFS: [string, string, string][] = [
    ["Crashes", "CRASHED", "a-crash"],
    ["Timeouts", "TIMED_OUT", "a-timeout"],
    ["OOMs", "OOM", "a-oom"],
];
const STAGE_LIMIT = 6;

export function Overview({ db, meta }: { db: Database; meta: Metadata }) {
    const stats = statusStats(db);
    const statsCount = (status: string) => statusCount(stats, status);
    const okPercent = ((statsCount("OK") * 100) / stats.total).toFixed(1);
    const runtime = stats.totalTime / meta.cpu_count;

    const detailsValues: [string, string][] = [
        ["Started", formatTimestamp(meta.start_time)],
        ["Runtime", `${formatClock(runtime)} - ${meta.cpu_count} CPUs`],
    ];
    for (const line of meta.notes?.split("\n") ?? []) {
        if (line.trim() !== "") {
            detailsValues.push(line.trim().split(":") as [string, string]);
        }
    }

    return (
        <>
            {/* KPI cards */}
            <div class="kpi-grid">
                <div class="kpi hero">
                    <div class="kpi-name">Success rate</div>
                    <div class="big">
                        <span class="num">{okPercent}</span>
                        <span class="pct">%</span>
                    </div>
                    <div class="kpi-sub">{`${statsCount("OK")} / ${stats.total} binaries`}</div>
                </div>
                {KPI_DEFS.map(([label, status, accent, href]) => (
                    <div
                        class={`kpi accent ${accent}${href ? " link" : ""}`}
                        onClick={href ? () => (window.location.href = href) : undefined}
                    >
                        <div class="kpi-top">
                            <span class="kpi-name">{label}</span>
                            {href ? <Icon name="chevron" class="chev" /> : null}
                        </div>
                        <div class="kpi-value">{statsCount(status)}</div>
                        <div class="kpi-sub kpi-value-accent">
                            {percent(statsCount(status), stats.total)}
                        </div>
                    </div>
                ))}
            </div>

            {/* Run composition bar */}
            <div class="card composition">
                <div class="comp-head">
                    <span class="title">Run composition</span>
                    <span class="meta">
                        {`${stats.total} binaries - ${formatHM(runtime)} on ${meta.cpu_count} CPUs`}
                    </span>
                </div>
                <div class="comp-bar">
                    {COMPOSITION.map(([label, status, color]) => (
                        <span
                            style={{
                                width: percent(statsCount(status), stats.total),
                                background: color,
                            }}
                            title={`${label}: ${statsCount(status)}`}
                        />
                    ))}
                </div>
                <div class="comp-legend">
                    {COMPOSITION.map(([label, status, color]) => (
                        <div class="item">
                            <span class="swatch" style={{ background: color }} />
                            {label}
                            <span class="v">{percent(statsCount(status), stats.total)}</span>
                        </div>
                    ))}
                </div>
            </div>

            <div class="section-head">
                <span class="title">Failure breakdown by component</span>
            </div>
            <div class="stage-cols">
                {STAGE_DEFS.map(([label, category, accent]) => (
                    <CrashComponentsCard
                        db={db}
                        meta={meta}
                        label={label}
                        category={category}
                        accent={accent}
                        headCount={statusCount(stats, category)}
                        limit={STAGE_LIMIT}
                    />
                ))}
            </div>

            {/* Notable binaries + run details */}
            <div class="two-grid">
                <div class="card">
                    <div class="card-label">Notable binaries</div>
                    <div class="notable-list">
                        {(meta.highlights || []).map((entry) => {
                            const rows = db.exec({
                                sql: `SELECT name FROM main ${entry.query} LIMIT 1`,
                                ...sqlOptions,
                            });
                            // Colour the badge by the status the highlight query
                            // filters on, rather than matching the free-form text.
                            const statusMatch = entry.query.match(/status\s*=\s*'(\w+)'/);
                            const badgeClass = statusMatch
                                ? `badge-${statusMatch[1]}`
                                : "badge-CRASHED";
                            const shortDesc = entry.description.replace(" binary", "");
                            return (
                                <div class="notable-item">
                                    <span class={`tag badge ${badgeClass}`}>{shortDesc}</span>
                                    {rows.length > 0 ? (
                                        <a
                                            href={`binary.html#${rows[0].name}`}
                                            title={rows[0].name as string}
                                        >
                                            {rows[0].name as string}
                                        </a>
                                    ) : (
                                        <span class="mono" style={{ color: "var(--muted)" }}>
                                            N/A
                                        </span>
                                    )}
                                </div>
                            );
                        })}
                    </div>
                </div>
                <div class="card">
                    <div class="card-label">Run details</div>
                    <div class="kv">
                        {detailsValues.map(([k, v]) => (
                            <>
                                <span class="k">{k}</span>
                                <span class="v">{v}</span>
                            </>
                        ))}
                    </div>
                </div>
            </div>
        </>
    );
}
