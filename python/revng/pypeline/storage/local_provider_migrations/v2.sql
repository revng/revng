--
-- This file is distributed under the MIT License. See LICENSE.md for details.
--

DELETE FROM objects;
DELETE FROM dependencies;
DELETE FROM custom_dependencies;

ALTER TABLE objects ADD COLUMN object_id_string TEXT NOT NULL;
