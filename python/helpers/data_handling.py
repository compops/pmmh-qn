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

"""Helpers for generating and importing data from/to models."""
import os.path
import pandas as pd
import numpy as np
import quandl


def import_data_quandl(model, handle, start_date, end_date, variable, api_key=None):
    """ Imports financial data from Quandl.

        Downloads data from Quandl and computes the log-returns in percent.
        The result is saved in the model object as the attributes obs and
        and no_obs.

        Args:
            model: object to store data in.
            handle: name at Quandl. (string)
            start_date: date to start extraction from (YYYY-MM-DD).
            start_date: date to end extraction at (YYYY-MM-DD).
            variable: name of column to use for computations.

        Returns:
           Nothing.

    """
    if api_key:
        quandl.ApiConfig.api_key = api_key
    else:
        # check for API key in file
        file_path = "helpers/quandl_key.nogit"
        if os.path.exists(file_path):
            with open(file_path, 'r') as myfile:
                api_key = myfile.read().replace('\n', '')
                quandl.ApiConfig.api_key = api_key
                print("Quandl API key loaded from file...")

    data = quandl.get(handle, start_date=start_date, end_date=end_date)
    log_returns = 100 * np.diff(np.log(data[variable]))
    log_returns = log_returns[~np.isnan(log_returns)]
    model.no_obs = len(log_returns) - 1
    obs = np.array(log_returns, copy=True).reshape((model.no_obs + 1, 1))
    model.obs = obs
    print("Loaded {} observations from Quandl.".format(model.no_obs))


def import_data(model, file_name):
    """ Imports data from file.

        Data is given as a csv file with first line as the labels observation,
        state and/or input. One data point per line. Only observation is
        required. The data is stored in the model object under the attributes
        obs, states, inputs and obs.

        Args:
            model: object to store data in.
            file_name: relative search path to csv file. (string)

        Returns:
           Nothing.

    """
    data_frame = pd.read_csv(file_name)

    if 'observation' in list(data_frame):
        if model.no_obs:
            obs = data_frame['observation'].values[0:(model.no_obs + 1)]
        else:
            obs = data_frame['observation'].values
            model.no_obs = len(obs) - 1
        obs = np.array(obs, copy=True).reshape((model.no_obs + 1, 1))
        model.obs = obs
    else:
        raise ValueError(
            "No observations in file, header must be observation.")

    if 'state' in list(data_frame):
        states = data_frame['state'].values[0:(model.no_obs + 1)]
        states = np.array(states, copy=True).reshape((model.no_obs + 1, 1))
        model.states = states

    if 'input' in list(data_frame):
        inputs = data_frame['input'].values[0:(model.no_obs + 1)]
        inputs = np.array(obs, copy=True).reshape((model.no_obs + 1, 1))
        model.inputs = inputs

    print("Loaded data from file: " + file_name + ".")


def import_paneldata_numpy(model, file_name_x, file_name_y):
    """ Imports panel data from a numpy data file.

        Data is given as two files created with numpy.save, one for the obs
        (individual, time_step) and one for the regressors (individual,
        time_step, regressor).

        Args:
            model: object to store data in.
            file_name_x: relative search path to npy file. (string)
            file_name_y: relative search path to npy file. (string)

        Returns:
           Nothing.

    """
    data_x = np.load(file_name_x)
    data_y = np.load(file_name_y)

    if model.no_individuals:
        data_x = data_x[:model.no_individuals, :, :]
        data_y = data_y[:model.no_individuals, :]

    model.states = {}
    model.states.update({'x': data_x})
    model.obs = data_y

    model.no_individuals = data_y.shape[0]
    model.no_obs = data_y.shape[1]
    model.no_regressors = data_x.shape[2]

    print("Loaded data from files: " + file_name_x + " and " + file_name_y + ".")
    print("No individuals: {}, no obs: {} and no regressors: {}.".format(
        data_x.shape[0], data_x.shape[1], data_x.shape[2]))


def generate_data(model, file_name=None):
    """ Generates data from model and saves it to file.

        Data is generated according to the model object and stored as the
        attributes obs and state. The data is saved to a csv file if a file
        name is provided.

        Args:
            model: object to store data in.
            file_name: relative path to csv file for saving data. (string)

        Returns:
           Nothing.

    """
    model.states = np.zeros((model.no_obs + 1, 1))
    model.obs = np.zeros((model.no_obs + 1, 1))
    model.states[0] = model.initial_state

    for i in range(1, model.no_obs + 1):
        model.states[i] = model.generate_state(model.states[i - 1], i)
        model.obs[i] = model.generate_obs(model.states[i], i)

    if file_name:
        data_frame = pd.DataFrame(data=np.hstack((model.states, model.obs)),
                                  columns=['state', 'observation'])
        data_frame.to_csv(file_name, index=False, header=True)
        print("Wrote generated data to file: " + file_name + ".")
