/*
 * This file is distributed under the MIT License. See LICENSE.md for details.
 */

import { Database, SQLite3Error } from "@sqlite.org/sqlite-wasm";
import DataTable, { Api, ConfigColumns } from "datatables.net-dt";
import { filesize } from "filesize";
import { ComponentChild, render } from "preact";
import { useEffect, useRef } from "preact/hooks";

import { sqlOptions } from "../utils/db";
import { formatTime } from "../utils/format";
import { PageConfig } from "../utils/pages";
import { ColumnDef, Metadata } from "../utils/types";
import { ActionsButton } from "./ActionMenu";
import { Icon } from "./Icons";

const DISPLAY_RENDERERS: Record<string, (data: any) => ComponentChild> = {
    time: (data: number) => formatTime(data),
    filesize: (data: number) => (data < 0 ? "-" : filesize(data, { base: 2 })),
    ellipsis: (data: string) => (data.length < 70 ? data : data.slice(0, 70) + "..."),
};

export interface TableContext {
    setSearch: (query: string) => void;
}

export function getColumns(meta: Metadata, ctx: TableContext, inDetail: boolean): ColumnDef[] {
    const cols: ColumnDef[] = [
        {
            name: "name",
            data: "name",
            title: "Name",
            cell: (data: string) => (
                <span class="cell-name" title={data}>
                    {data}
                </span>
            ),
            useCellForDetail: false,
        },
        {
            name: "elapsed_time",
            data: "elapsed_time",
            title: "Time",
            className: "dt-right mono",
            cell: (data: number) => formatTime(data),
        },
        { name: "exit_code", data: "exit_code", title: "Exit", className: "dt-right mono" },
        {
            name: "status",
            data: "status",
            title: "Status",
            cell: (data: string) => <span class={`badge badge-${data}`}>{data}</span>,
        },
        {
            name: "stacktrace_id",
            data: "stacktrace_id",
            title: "Stacktrace",
            orderable: false,
        },
    ];

    for (const elem of meta.extra_columns || []) {
        const col: ColumnDef = { name: elem.name, data: elem.name, title: elem.label };
        const renderer = elem.renderer !== undefined ? DISPLAY_RENDERERS[elem.renderer] : undefined;
        if (renderer !== undefined) {
            col.cell = (data: any) => renderer(data);
        }
        col.className = `dt-${elem.align ?? "left"} mono`;
        cols.push(col);
    }

    cols.push({
        name: "actions",
        title: "Actions",
        orderable: false,
        searchable: false,
        className: "dt-right",
        cell: (_data: any, row: any) => <ActionsButton row={row} meta={meta} inDetail={inDetail} />,
    });

    return cols;
}

function renderToNode(child: ComponentChild): Node | string {
    if (typeof child === "string") {
        return child;
    }
    const host = document.createElement("div");
    render(child, host);
    return host.firstChild ?? host;
}

function dtColumns(defs: ColumnDef[]): ConfigColumns[] {
    return defs.map((def) => {
        const col: ConfigColumns = { name: def.name, title: def.title };
        if (def.data !== undefined) col.data = def.data;
        if (def.className !== undefined) col.className = def.className;
        if (def.orderable !== undefined) col.orderable = def.orderable;
        if (def.searchable !== undefined) col.searchable = def.searchable;
        if (def.cell !== undefined) {
            const cell = def.cell;
            col.render = (data: any, type: string, row: any) => {
                if (type !== "display") return def.data ? data : "";
                return renderToNode(cell(data, row));
            };
        }
        return col;
    });
}

class SearchState {
    public query: string;
    public sql: boolean;

    constructor(query: string, sql: boolean) {
        this.query = query;
        this.sql = sql;
    }

    toString() {
        return btoa(JSON.stringify(this));
    }

    static fromString(data: string): SearchState {
        const obj = JSON.parse(atob(data));
        return new SearchState(obj.query, obj.sql);
    }
}

export function DataTableView({
    db,
    meta,
    cfg,
}: {
    db: Database;
    meta: Metadata;
    cfg: PageConfig;
}) {
    const cardRef = useRef<HTMLDivElement>(null);
    const toggleRef = useRef<HTMLLabelElement>(null);
    const searchRef = useRef<HTMLInputElement>(null);
    const checkboxRef = useRef<HTMLInputElement>(null);

    useEffect(() => {
        const host = cardRef.current!;
        const table = document.createElement("table");
        table.className = "compact";
        table.style.width = "100%";
        host.append(table);

        let baseQuery = "SELECT * FROM main WHERE 1=1";
        if (cfg.status) {
            baseQuery += ` AND status = '${cfg.status}'`;
        }

        const getData = (sql: string) => db.exec({ sql, ...sqlOptions });
        const baseData = getData(baseQuery);

        const searchNode = renderToNode(
            <div class="table-search">
                <div class="searchbox">
                    <Icon name="search" class="icon"></Icon>
                    <input
                        type="search"
                        ref={searchRef}
                        placeholder="Search name, stacktrace id..."
                        onInput={() => {
                            if (!checkboxRef.current!.checked) {
                                runSearch(updateState());
                            }
                        }}
                        onKeyUp={(ev) => {
                            if (
                                checkboxRef.current!.checked &&
                                (ev.key === "Enter" || ev.keyCode === 13)
                            ) {
                                runSearch(updateState());
                            }
                        }}
                    ></input>
                </div>
                <label
                    class="sql-toggle"
                    ref={toggleRef}
                    title="Toggle raw SQL WHERE-clause search"
                >
                    <input
                        type="checkbox"
                        ref={checkboxRef}
                        onInput={(ev) => {
                            const checkbox = ev.target as HTMLInputElement;
                            toggleRef.current!.classList.toggle("on", checkbox.checked);
                            if (checkbox.checked) {
                                dt.search("");
                            } else {
                                dt.clear();
                                dt.rows.add(baseData);
                            }
                            dt.draw();
                            updateState();
                        }}
                    ></input>
                    SQL
                </label>
            </div>
        );

        const ctx: TableContext = { setSearch: () => {} };
        const dt: Api<any> = new DataTable(table, {
            data: baseData,
            columns: dtColumns(getColumns(meta, ctx, false)),
            order: meta.ordering as any,
            pageLength: 10,
            layout: {
                topStart: (() =>
                    renderToNode(
                        <div class="table-title">
                            {baseData.length} {cfg.rowNoun}
                        </div>
                    )) as unknown as null,
                topEnd: (() => searchNode) as unknown as null,
                bottomStart: "info",
                bottomEnd: "paging",
            },
        });

        function runSearch(state: SearchState) {
            if (state.sql) {
                let data;
                try {
                    data = getData(`${baseQuery} AND (${state.query})`);
                } catch (e) {
                    if (e instanceof SQLite3Error) {
                        return;
                    }
                    throw e;
                }
                dt.clear();
                dt.rows.add(data);
            } else {
                dt.search(state.query);
            }
            dt.draw();
        }

        function updateState(): SearchState {
            const state = new SearchState(searchRef.current!.value, checkboxRef.current!.checked);
            window.location.hash = `#${state.toString()}`;
            return state;
        }

        ctx.setSearch = (query: string) => {
            checkboxRef.current!.checked = false;
            toggleRef.current!.classList.remove("on");
            searchRef.current!.value = query;
            runSearch(updateState());
        };

        // Restore a persisted search from the hash
        if (window.location.hash.startsWith("#")) {
            try {
                const state = SearchState.fromString(window.location.hash.slice(1));
                checkboxRef.current!.checked = state.sql;
                toggleRef.current!.classList.toggle("on", state.sql);
                searchRef.current!.value = state.query;
                runSearch(state);
            } catch {
                /* ignore malformed hash */
            }
        }

        return () => {
            dt.destroy();
            host.innerHTML = "";
        };
    }, []);

    return <div class="table-card" ref={cardRef} />;
}
