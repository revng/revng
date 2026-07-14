/*
 * This file is distributed under the MIT License. See LICENSE.md for details.
 */

// ---- Per-status/per-page configuration ----------------------------------

export interface PageConfig {
    page: string;
    title: string;
    status?: string; // filter applied to the `main` table
    category?: string; // crash_components category, for stage bars/flamegraphs
    fgPrefix?: string; // flamegraph file prefix
    accent: string; // css accent class, e.g. a-crash
    rowNoun: string; // "crashed binaries"
    fgEnd?: string; // "crash", "timeout", "OOM" location wording
}

export const PAGES: Record<string, PageConfig> = {
    overview: { page: "overview", title: "Overview", accent: "a-crash", rowNoun: "binaries" },
    failures: {
        page: "failures",
        title: "Failures",
        status: "FAILED",
        accent: "a-fail",
        rowNoun: "failed binaries",
    },
    crashes: {
        page: "crashes",
        title: "Crashes",
        status: "CRASHED",
        category: "CRASHED",
        fgPrefix: "failures",
        accent: "a-crash",
        rowNoun: "crashed binaries",
        fgEnd: "crash",
    },
    timeouts: {
        page: "timeouts",
        title: "Timeouts",
        status: "TIMED_OUT",
        category: "TIMED_OUT",
        fgPrefix: "timeouts",
        accent: "a-timeout",
        rowNoun: "timed-out binaries",
        fgEnd: "timeout",
    },
    ooms: {
        page: "ooms",
        title: "OOMs",
        status: "OOM",
        category: "OOM",
        fgPrefix: "ooms",
        accent: "a-oom",
        rowNoun: "out-of-memory binaries",
        fgEnd: "OOM",
    },
    successes: {
        page: "successes",
        title: "Successes",
        status: "OK",
        accent: "a-ok",
        rowNoun: "successful binaries",
    },
    raw_data: { page: "raw_data", title: "Raw Data", accent: "a-crash", rowNoun: "binaries" },
};

// Tabs appear in the order the pages are declared in PAGES
export const NAV_ORDER = Object.keys(PAGES);

export function pageHref(key: string): string {
    return key === "overview" ? "index.html" : `${key}.html`;
}
