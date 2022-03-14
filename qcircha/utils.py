# This code is part of qcircha.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

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

            # Removing trash numbers at the end of some circuit names
            if len(model_data['fmap'].split('-')) > 1:
                #print("Changing name! [can ignore]")
                model_data['fmap'] = model_data['fmap'].split('-')[0]

            if len(model_data['var_ansatz'].split('-')) > 1:
                #print("Changing name! [can ignore]")
                model_data['var_ansatz'] = model_data['var_ansatz'].split('-')[0]
                        
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