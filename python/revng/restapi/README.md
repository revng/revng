# rev.ng REST API

## How to start a dev server

After `orc install revng`:

To start a dev server from orchestra root:
```shell
orc shell
FLASK_APP=revng.restapi FLASK_ENV=development flask run
```

To start a dev server from the build directory:
```shell
# `-c revng` is important as it populates BUILD_DIR!
orc shell -c revng
export PYTHONPATH="$BUILD_DIR/lib/python"
FLASK_APP=revng.restapi FLASK_ENV=development flask run
```
