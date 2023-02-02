from os.path import dirname, abspath


def project_dir():
    """
    :return:  current project directory path
    """
    return dirname(dirname(dirname(abspath(__file__))))
