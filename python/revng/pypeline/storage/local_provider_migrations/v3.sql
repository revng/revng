--
-- This file is distributed under the MIT License. See LICENSE.md for details.
--

-- Adds `last_seen_model`, the serialized copy of the model as of the last
-- set_model. It lets us diff against the on-disk model and purge only the
-- affected objects when the model file is edited behind our back, instead of
-- discarding all the caches. The caches are cleared here because they cannot
-- be reconciled with a baseline that did not exist before this migration.

DELETE FROM objects;
DELETE FROM dependencies;
DELETE FROM custom_dependencies;

ALTER TABLE project ADD COLUMN last_seen_model BLOB;
