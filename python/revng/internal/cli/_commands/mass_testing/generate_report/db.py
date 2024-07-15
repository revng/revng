#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from dataclasses import dataclass
from pathlib import Path
from sqlite3 import Cursor, connect
from typing import Callable, Dict, Iterable, List, Tuple, Union

from .meta import GlobalMeta
from .test_directory import TestDirectory

DbTypes = Union[str, bool, int, float]
ColumnDefinition = Tuple[str, Callable[[TestDirectory], DbTypes]]


@dataclass
class Schema:
    columns: List[ColumnDefinition]
    generated_columns: List[str]

    def clone(self):
        return Schema(self.columns.copy(), self.generated_columns.copy())

    def all_columns(self):
        return [r[0] for r in self.columns] + self.generated_columns


TYPES_MAPPING = {
    "int": "INTEGER",
    "float": "REAL",
    "str": "TEXT",
    "bool": "INTEGER",
}

DEFAULT_COLUMNS = Schema(
    [
        ("name TEXT PRIMARY KEY NOT NULL", lambda td: td.name),
        ("input_name TEXT NOT NULL", lambda td: td.input_name),
        ("elapsed_time REAL NOT NULL", lambda td: td.elapsed_time),
        ("exit_code INTEGER NOT NULL", lambda td: td.exit_code),
        ("status TEXT NOT NULL", lambda td: td.status),
        ("has_input INTEGER NOT NULL", lambda td: td.has_input()),
        ("has_trace INTEGER NOT NULL", lambda td: td.has_trace()),
        ("stacktrace_id TEXT NOT NULL", lambda td: td.stacktrace_id()),
    ],
    [],
)


def create_schema(meta: GlobalMeta | None) -> Schema:
    result = DEFAULT_COLUMNS.clone()
    if meta is None:
        return result

    for entry in meta.extra_columns:
        if entry.generator is not None:
            result.generated_columns.append(
                f"{entry.name} {TYPES_MAPPING[entry.type]} NOT NULL "
                + f"GENERATED ALWAYS AS ({entry.generator}) STORED",
            )
        else:

            def get_meta(td: TestDirectory, name=entry.name):
                return td.get_meta(name)

            result.columns.append((f"{entry.name} {TYPES_MAPPING[entry.type]} NOT NULL", get_meta))

    return result


def create_main_table(cursor: Cursor, schema: Schema, items: Iterable[TestDirectory]):
    query = "CREATE TABLE IF NOT EXISTS main(" + ", ".join(schema.all_columns()) + ")"
    cursor.execute(query)
    query = "INSERT INTO main VALUES(" + ", ".join("?" for r in schema.columns) + ")"
    cursor.executemany(query, ([r[1](item) for r in schema.columns] for item in items))


def create_components_table(cursor: Cursor, items: Dict[str, Dict[str, int]]):
    query = (
        "CREATE TABLE IF NOT EXISTS crash_components"
        + "(name TEXT NOT NULL, category TEXT NOT NULL, count INTEGER NOT NULL);"
    )
    cursor.execute(query)
    res: List[Tuple[str, str, int]] = []
    for category, data in items.items():
        for name, value in data.items():
            res.append((name, category, value))
    cursor.executemany("INSERT INTO crash_components VALUES(?, ?, ?)", res)


def create_and_populate(
    path: str | Path,
    items: Iterable[TestDirectory],
    crash_counts: Dict[str, Dict[str, int]],
    meta: GlobalMeta | None,
):
    schema = create_schema(meta)
    conn = connect(str(path))
    cursor = conn.cursor()
    create_main_table(cursor, schema, items)
    create_components_table(cursor, crash_counts)
    conn.commit()
    conn.close()
