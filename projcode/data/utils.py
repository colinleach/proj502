import yaml
from pathlib import Path
from os import path


def read_params():
    """
    Needs the gzoo.yaml parameter file to be in the current directory

    :return: parameter dictionary
    """

    filename = Path(path.expanduser("~")) / '.gzoo.yaml'
    with open(filename) as file:
        params = yaml.full_load(file)
        return params
