--
-- This file is distributed under the MIT License. See LICENSE.md for details.
--

CREATE TABLE IF NOT EXISTS project(
    project_id                   UUID PRIMARY KEY,
    last_fetch                   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_object_save             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_model_save              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    epoch                        INT NOT NULL DEFAULT 0,
    pipeline_description_hash    TEXT,
    version                      TEXT,
    model                        BYTEA
);

CREATE TABLE IF NOT EXISTS objects(
    project_id           UUID REFERENCES project(project_id),
    savepoint_id         INT NOT NULL,
    container_id         TEXT NOT NULL,
    configuration_hash   TEXT NOT NULL,
    object_id            BYTEA NOT NULL,
    object_id_string     TEXT NOT NULL,
    content              BYTEA NOT NULL,
    PRIMARY KEY (project_id, savepoint_id, container_id, configuration_hash, object_id)
);

CREATE INDEX IF NOT EXISTS project_id_on_object ON objects(project_id);
CREATE INDEX IF NOT EXISTS savepoint_id_on_object ON objects(savepoint_id);
CREATE INDEX IF NOT EXISTS container_id_on_object ON objects(container_id);
CREATE INDEX IF NOT EXISTS configuration_hash_on_object ON objects(configuration_hash);
CREATE INDEX IF NOT EXISTS object_id_on_object ON objects(object_id);

CREATE TABLE IF NOT EXISTS dependencies(
    project_id           UUID REFERENCES project(project_id),
    savepoint_id_start   INT NOT NULL,
    savepoint_id_end     INT NOT NULL,
    container_id         TEXT NOT NULL,
    configuration_hash   TEXT NOT NULL,
    object_id            BYTEA NOT NULL,
    model_path           TEXT NOT NULL,
    PRIMARY KEY (project_id, savepoint_id_start, savepoint_id_end, container_id,
                 configuration_hash, object_id, model_path)
);

CREATE INDEX IF NOT EXISTS project_id_on_dependencies ON dependencies(project_id);
CREATE INDEX IF NOT EXISTS model_path_on_dependencies ON dependencies(model_path);

CREATE TABLE IF NOT EXISTS custom_dependencies(
    project_id           UUID REFERENCES project(project_id),
    pipe_id              INT NOT NULL,
    configuration_hash   TEXT NOT NULL,
    argument_index       INT NOT NULL,
    object_id            BYTEA NOT NULL,
    data                 BYTEA NOT NULL,
    PRIMARY KEY (project_id, pipe_id, configuration_hash, argument_index, object_id)
);

CREATE TABLE IF NOT EXISTS locks(
    project_id           UUID REFERENCES project(project_id),
    lock_id              UUID NOT NULL,
    lock_type            INT NOT NULL,
    creation_timestamp   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    refresh_timestamp    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    in_use               BOOLEAN NOT NULL DEFAULT FALSE,
    PRIMARY KEY (project_id, lock_id)
);

CREATE INDEX IF NOT EXISTS project_status_index_on_locks
    ON locks(project_id, lock_type);
CREATE INDEX IF NOT EXISTS locks_refresh_index_on_locks
    ON locks(project_id, refresh_timestamp);

CREATE TABLE IF NOT EXISTS pipeline_descriptions(
    hash         TEXT PRIMARY KEY,
    content      BYTEA NOT NULL
);

CREATE TABLE IF NOT EXISTS file_storage(
    project_id   UUID REFERENCES project(project_id),
    hash         TEXT NOT NULL,
    content      BYTEA NOT NULL,
    PRIMARY KEY (project_id, hash)
);
