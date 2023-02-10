from os.path import dirname, abspath

import numpy as np


def print_array_details(a: np.ndarray):
    print(
        f"""
        ndim: ${a.ndim},
        shape: ${a.shape},
        """
    )


def file_from_data_dir(filename: str):
    return dirname(project_dir()) + "/data/" + filename


def project_dir():
    return dirname(dirname(dirname(abspath(__file__))))
