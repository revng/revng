--
-- This file is distributed under the MIT License. See LICENSE.md for details.
--

CREATE TABLE IF NOT EXISTS project(
    id              TEXT PRIMARY KEY CHECK (id = 0),
    last_fetch         REAL NOT NULL,
    last_object_save   REAL NOT NULL,
    last_model_save    REAL NOT NULL,
    epoch           INT NOT NULL,
    pipeline_hash   TEXT,
    version         TEXT,
    model_hash      TEXT,
    model_mtime     REAL
) STRICT;

-- Add the only row to the DB, this allows to simplify the logic since this row
-- is always guaranteed to exist and avoids having code that creates it
-- opportunistically
INSERT OR IGNORE INTO project
(id, epoch, last_fetch, last_object_save, last_model_save)
VALUES (0, 0, 0.0, 0.0, 0.0);

CREATE TABLE IF NOT EXISTS objects(
    savepoint_id         INT NOT NULL,
    container_id         TEXT NOT NULL,
    configuration_hash   TEXT NOT NULL,
    object_id            BLOB NOT NULL,
    content              BLOB NOT NULL,
    PRIMARY KEY (savepoint_id, container_id, configuration_hash, object_id)
) STRICT;

CREATE INDEX IF NOT EXISTS savepoint_id_on_object ON objects(savepoint_id);
CREATE INDEX IF NOT EXISTS container_id_on_object ON objects(container_id);
CREATE INDEX IF NOT EXISTS configuration_hash_on_object ON objects(configuration_hash);
CREATE INDEX IF NOT EXISTS object_id_on_object ON objects(object_id);

CREATE TABLE IF NOT EXISTS dependencies(
    savepoint_id_start   INT NOT NULL,
    savepoint_id_end     INT NOT NULL,
    container_id         TEXT NOT NULL,
    configuration_hash   TEXT NOT NULL,
    object_id            BLOB NOT NULL,
    model_path           TEXT NOT NULL,
    PRIMARY KEY (savepoint_id_start, savepoint_id_end, container_id,
                 configuration_hash, object_id, model_path)
) STRICT;

CREATE INDEX IF NOT EXISTS model_path_on_dependencies ON dependencies(model_path);

CREATE TABLE IF NOT EXISTS custom_dependencies(
    pipe_id              INT NOT NULL,
    configuration_hash   TEXT NOT NULL,
    argument_index       INT NOT NULL,
    object_id            BLOB NOT NULL,
    data                 BLOB NOT NULL,
    PRIMARY KEY (pipe_id, configuration_hash)
) STRICT;

CREATE INDEX IF NOT EXISTS custom_dependencies_index
    ON custom_dependencies(pipe_id, configuration_hash);
