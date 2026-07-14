/*
 * This file is distributed under the MIT License. See LICENSE.md for details.
 */

import sqlite3InitModule, { Database, Sqlite3Static } from "@sqlite.org/sqlite-wasm";

import { Metadata, StageEntry, StatusStats } from "./types";

// We inject some variables into `window`, for easier debugging
declare global {
    interface Window {
        _sqlite3?: Sqlite3Static;
        db?: Database;
    }
}

export const sqlOptions = { rowMode: "object", returnValue: "resultRows" } as const;

async function initSqlite3(): Promise<Sqlite3Static> {
    if (window._sqlite3 !== undefined) {
        return window._sqlite3;
    } else {
        const sqlite3 = await sqlite3InitModule({
            print: console.log,
            printErr: console.error,
        });
        window._sqlite3 = sqlite3;
        return sqlite3;
    }
}

export async function loadDBFromURL(url: string | URL): Promise<Database> {
    const sqlite3 = await initSqlite3();
    const req = await fetch(url);
    const buffer = await req.arrayBuffer();
    const db_buffer = sqlite3.wasm.allocFromTypedArray(buffer);
    const db = new sqlite3.oo1.DB();
    const rc = sqlite3.capi.sqlite3_deserialize(
        db,
        "main",
        db_buffer,
        buffer.byteLength,
        buffer.byteLength,
        sqlite3.capi.SQLITE_DESERIALIZE_FREEONCLOSE
    );
    db.checkRc(rc);
    return db;
}

export function sleep(time: number): Promise<void> {
    return new Promise((res) => setTimeout(() => res(undefined), time));
}

// Per-status counts (plus totals) in a single grouped scan, so callers do not
// need to know the set of statuses ahead of time.
export function statusStats(db: Database): StatusStats {
    const rows = db.exec({
        sql: "SELECT status, COUNT(*) AS n, SUM(elapsed_time) AS t FROM main GROUP BY status",
        ...sqlOptions,
    });
    const counts: Record<string, number> = {};
    let total = 0;
    let totalTime = 0;
    for (const r of rows) {
        counts[r.status as string] = r.n as number;
        total += r.n as number;
        totalTime += (r.t as number) || 0;
    }
    return { total, totalTime, counts };
}

export function statusCount(s: StatusStats, status: string): number {
    return s.counts[status] ?? 0;
}

export function distinctSignatures(db: Database, status: string): number {
    const rows = db.exec({
        sql: "SELECT COUNT(DISTINCT stacktrace_id) AS n FROM main WHERE status = ? AND stacktrace_id != ''",
        bind: [status],
        ...sqlOptions,
    });
    return (rows[0]?.n as number) ?? 0;
}

export function stageEntries(db: Database, category: string, filter: string): StageEntry[] {
    const rows = db.exec({
        sql: "SELECT name, count FROM crash_components WHERE category = ? AND filter = ?",
        bind: [category, filter],
        ...sqlOptions,
    });
    const entries = rows.map((r) => ({ name: r.name as string, count: r.count as number }));
    entries.sort((a, b) => b.count - a.count);
    return entries;
}

export function categoryFilters(meta: Metadata, category: string): [string, string][] {
    const result: [string, string][] = [["all", "all"]];
    for (const entry of meta.crash_components_filters ?? []) {
        if (entry.category === category) {
            result.push([entry.suffix, entry.label]);
        }
    }
    return result;
}
