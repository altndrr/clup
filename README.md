# IROS 2022

## Setup

### Requirements.

 - cuda
 - poetry

### Installation

To create the environment and install the dependencies, run the following:

```sh
poetry install
```

#### CUDA 11.3

For GPUs requiring cu11.3, overwrite the installation with pip

```sh
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113 --force
```

## Development tools

```sh
black src                                       # format code
ipython                                         # open an interactive shell
isort src                                       # sort imports
mypy src                                        # check static types
pylint src                                      # check lint quality
sphinx-apidoc -o docs/source/ .                 # create rst from docstring
sphinx-build -b html docs/source/ docs/build/   # create html pages from rst
```
