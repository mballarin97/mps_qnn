"""
General utility functions
"""

import sys
import os

import os
import sys
import json

def list_data(path=None):
    idx_list = []
    for file in os.listdir(path):
        if file.endswith(".json"):
            idx = file[:-5]  # delete ".json"
            idx_list.append(idx)

    return idx_list

def gather_data(key = None, value = None, path = None):
    """
    Find training runs with a specific param_perturbation.
    """

    idx_list = []
    for file in os.listdir(path):
        if file.endswith(".json"):
            with open(os.path.join(path, file)) as datafile:
                model_data = json.load(datafile)
                        
            if all([model_data[k] == v for k, v in zip(key, value)]):
                idx = file[:-5]  # delete ".json"
                idx_list.append(idx)

    return idx_list

def logger(data):
    """
    Printing data.

    Parameters
    ----------
    data : dict
        Simulation details

    Returns
    -------
    None
    """

    print("===============================")
    print("SIMULATION DETAILS")
    for k,v in zip(data.keys(), data.values()):
        print(f"{k} = {v}", end="\n")
    print("===============================")

class HiddenPrints:
    """
    Describe 
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def removekey(dictionary, keys):
    """
    Remove the keys from a dictionary dictionary

    Parameters
    ----------
    dictionary : dict
        Dictionary from where we remove the keys
    keys : array-like
        keys to be removed

    Returns
    -------
    dict
        dictionary with the removed keys
    """
    r = dict(dictionary)
    for key in keys:
        del r[key]
    return r