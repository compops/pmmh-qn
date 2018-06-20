###############################################################################
#    Correlated pseudo-marginal Metropolis-Hastings using
#    quasi-Newton proposals
#    Copyright (C) 2018  Johan Dahlin < uni (at) johandahlin [dot] com >
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
###############################################################################

"""Helpers for manipulating the file system."""
import json
import gzip
import os

import numpy as np


def ensure_dir(file_name):
    """ Check if dirs for outputs exists, otherwise create them

        Args:
            file_name: relative search path to file (string).

        Returns:
           Nothing.

    """
    directory = os.path.dirname(file_name)
    if not os.path.exists(directory):
        os.makedirs(directory)


def write_to_json(data, output_path, sim_name, output_type, as_gzip=True):
    """ Writes result of state/parameter estimation to file.

        Writes results in the form of a dictionary to file as JSON.

        Args:
            data: dict to store to file.
            output_path: relative file path to dir to store file in.
                         Without / at the end.
            sim_name: name of simulation (determines search path).
                      Without / at the end.
            output_type: name of the type of output (determines file name)

        Returns:
           Nothing.

    """
    # Convert NumPy arrays to lists
    for key in data:
        if isinstance(data[key], np.ndarray):
            data[key] = data[key].tolist()

    # Check if the directories exists and write data as json
    file_name = output_path + '/' + sim_name + '/' + output_type
    ensure_dir(file_name)

    if as_gzip:
        with gzip.GzipFile(file_name + '.gz', 'w') as fout:
            json_str = json.dumps(data, allow_nan=True)
            json_bytes = json_str.encode('utf-8')
            fout.write(json_bytes)
    else:
        with open(file_name, 'w') as f:
            json.dump(data, f, ensure_ascii=False)

    print("Wrote results to: " + file_name + ".")


def write_to_feather(data, output_path, sim_name, output_type, as_gzip=True):

    # Check if the directories exists and write data as feather
    file_name = output_path + '/' + sim_name + '/' + output_type + '.feather'
    ensure_dir(file_name)
    data.to_feather(file_name)
    print("Wrote results to: " + file_name + ".")
