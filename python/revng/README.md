# rev.ng python package

This python package provides classes that can be used to represent, serialize and deserialize rev.ng models.

It also contains other modules related to various revng CLI tools.

## How to install

You probably want to install this package in a virtualenv:

```shell
python -m venv <virtualenv_dir>
source <virtualenv_dir>/bin/activate
```

### Installation

To build and install from source:

```shell
pip install .
```

### Creating redistributable packages

This command will create a redistributable package in `dist`:

```shell
python setup.py bdist_wheel
```

The package can be installed using `pip`:

```bash
pip install dist/revng*.whl
```
