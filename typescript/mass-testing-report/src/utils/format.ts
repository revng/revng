/*
 * This file is distributed under the MIT License. See LICENSE.md for details.
 */

export function truncate(num: number, digits: number): number {
    return Math.trunc(num * 10 ** digits) / 10 ** digits;
}

export function percent(num: number, total: number): string {
    return `${truncate((num * 100) / total, 2)}%`;
}

function padNumber(num: number, amount: number): string {
    return `${num}`.padStart(amount, "0");
}

function splitHMS(seconds: number): { h: number; m: number; s: number } {
    return {
        h: Math.floor(seconds / 3600),
        m: Math.floor((seconds / 60) % 60),
        s: Math.floor(seconds % 60),
    };
}

// Per-run elapsed time as MM:ss.mmm
export function formatTime(seconds: number): string {
    const minutes = padNumber(Math.floor(seconds / 60), 2);
    const fmtSeconds = padNumber(Math.floor(seconds % 60), 2);
    const milliseconds = seconds.toFixed(3).split(".", 2)[1];
    return `${minutes}:${fmtSeconds}.${milliseconds}`;
}

// hh:mm:ss without milliseconds
export function formatClock(seconds: number): string {
    const { h, m, s } = splitHMS(seconds);
    return `${padNumber(h, 2)}:${padNumber(m, 2)}:${padNumber(s, 2)}`;
}

// "41h 57m"
export function formatHM(seconds: number): string {
    const { h, m } = splitHMS(seconds);
    return `${h}h ${padNumber(m, 2)}m`;
}

// From a UNIX timestamp (seconds), "2026-07-03 15:48:39 UTC"
export function formatTimestamp(unixSeconds: number): string {
    return new Date(unixSeconds * 1000).toISOString().replace("T", " ").slice(0, 19) + " UTC";
}
