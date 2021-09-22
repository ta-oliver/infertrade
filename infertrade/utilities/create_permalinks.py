import os
from infertrade.algos.community.allocations import create_infertrade_export_allocations


def make_permalinks_py_file():
    file_dir = os.getcwd()
    file_name = "permalinks.py"
    file_path = file_dir + file_name

    with open(file_path, 'w') as obj:
        obj.write(data=create_infertrade_export_allocations())
