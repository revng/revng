# How to add a migration to the local storage provider

1. Look at the current schema
2. Create a file named `vXXXX.sql`, the files need to be in order, e.g. `v1 v2 ... v9 v10` etc.
   The migration machinery will check that there are no holes
3. Add the needed migrations, these will probably be comprised of:
   * Some `DELETE FROM` statements to clear some tables
   * One or more `ALTER TABLE` statements

## General logic for the migrations

1. Start with the `ALTER TABLE` statements you want to make
2. Do they introduce nullable columns? If so no additional operations are needed
3. If they need non-nullable columns can the data be derived from the existing
   data in the database? Note that the pipebox is not available for migrations
   so these transformations need to be SQL-only.
   If not then it's ok to drop data from the following tables:
   * `objects`
   * `dependencies`
   * `custom_dependencies`
4. Usually you want to drop all three tables, they are inter-dependent so the
   data from one missing usually leads to corrupted data on the other
