import yaml


def read_params():
    """
    Needs the gzoo.yaml parameter file to be in the current directory

    :return: parameter dictionary
    """

    filename = '../gzoo.yaml'
    with open(filename) as file:
        params = yaml.full_load(file)
        return params
