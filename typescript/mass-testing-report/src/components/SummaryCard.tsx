/*
 * This file is distributed under the MIT License. See LICENSE.md for details.
 */

import { percent } from "../utils/format";
import { PageConfig } from "../utils/pages";

export function SummaryCard({
    cfg,
    count,
    total,
    note,
}: {
    cfg: PageConfig;
    count: number;
    total: number;
    note?: string;
}) {
    return (
        <div class={`summary-card ${cfg.accent}`}>
            <div class="row">
                <span class="dot" />
                <span class="name">{cfg.title}</span>
            </div>
            <div class="big">
                <span class="num">{count}</span>
                <span class="pct">{percent(count, total)}</span>
            </div>
            {note ? <div class="note">{note}</div> : null}
        </div>
    );
}
