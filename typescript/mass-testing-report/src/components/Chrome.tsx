/*
 * This file is distributed under the MIT License. See LICENSE.md for details.
 */

import { ComponentChildren } from "preact";

import { formatTimestamp } from "../utils/format";
import { NAV_ORDER, PAGES, PageConfig, pageHref } from "../utils/pages";
import { Metadata } from "../utils/types";

export function Chrome({
    meta,
    cfg,
    children,
}: {
    meta: Metadata;
    cfg: PageConfig;
    children: ComponentChildren;
}) {
    return (
        <div class="panel">
            <div class="topbar">
                <div class="brand">
                    <span class="brand-title">Mass Testing</span>
                </div>
                <div class="badges">
                    <span class="pill">Report - {formatTimestamp(meta.start_time)}</span>
                </div>
            </div>
            <div class={`tabs ${cfg.accent}`}>
                {NAV_ORDER.map((key) => (
                    <a href={pageHref(key)} class={key === cfg.page ? "active" : ""}>
                        {PAGES[key].title}
                    </a>
                ))}
            </div>
            <div class="content">{children}</div>
        </div>
    );
}
