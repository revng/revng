# rev.ng python package

This python package provides the following features:

- classes that can be used to represent, serialize and deserialize rev.ng models
- a converter of IDA databases (IDB and I64) to rev.ng models

## How to install

You probably want to install this package in a virtualenv:

```shell
python -m venv <virtualenv_dir>
source <virtualenv_dir>/bin/activate
```

### Prerequisites

You will need to install the rev.ng `python-idb` fork.

TODO: document where to find it

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
