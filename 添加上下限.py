import requests
import torch
import time  # to measure the computation time
import gym
from gym import spaces, core
import numpy as np
import random
import pandas as pd
import math
import os
import torch.nn as nn
import datetime
from gym.utils import seeding

from stable_baselines3 import TD3
from stable_baselines3 import PPO, SAC
from stable_baselines3 import DDPG
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
import stable_baselines3
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import matplotlib.pyplot as plt
import seaborn as sns

from collections import OrderedDict
from scipy import interpolate
from pprint import pformat


class BoptestGymEnv(gym.Env):
    '''
    BOPTEST Environment that follows gym interface.
    This environment allows the interaction of RL agents with building
    emulator models from BOPTEST.

    '''

    metadata = {'render.modes': ['console']}

    def __init__(self,
                 url='http://127.0.0.1:5000',
                 actions=['fcu_oveTSup_u', 'fcu_oveFan_u'],
                 observations={'zon_reaTRooAir_y': (280., 310.),  # 室内温度
                               'zon_weaSta_reaWeaHDirNor_y': (0., 982.62),  # 直接辐射量
                               'zon_weaSta_reaWeaHGloHor_y': (0., 1027.25),  # 全球水平太阳辐射
                               'zon_weaSta_reaWeaRelHum_y': (0., 1),  # 相对湿度
                               'zon_weaSta_reaWeaTDryBul_y': (248., 310),  # 外部干球温度
                               'zon_weaSta_reaWeaTWetBul_y': (248., 294),  # 外部湿球温度
                               'zon_weaSta_reaWeaNTot_y': (0., 1.),  # 天空覆盖
                               'HDirNor': (0., 982.62),  # 直接辐射量预测
                               'HGloHor': (0., 1027.25),  # 全球水平太阳辐射预测
                               'relHum': (0., 1),  # 相对湿度预测
                               'TDryBul': (248., 310),  # 外部干球温度预测
                               'TWetBul': (248., 294),  # 外部湿球温度预测
                               'nTot': (0., 1.),  # 天空覆盖预测
                               'PriceElectricPowerDynamic': (0.04440, 0.13814),  # 电价
                                'UpperSetp[1]': (280., 310.),
                               'LowerSetp[1]': (280., 310.),
                               },

                 min_max_info={'zon_reaTRooAir_y': (280., 310.),  # 室内温度
                               'zon_weaSta_reaWeaHDirNor_y': (0., 982.62),  # 直接辐射量
                               'zon_weaSta_reaWeaHGloHor_y': (0., 1027.25),  # 全球水平太阳辐射
                               'zon_weaSta_reaWeaRelHum_y': (0., 1),  # 相对湿度
                               'zon_weaSta_reaWeaTDryBul_y': (248., 310),  # 外部干球温度
                               'zon_weaSta_reaWeaTWetBul_y': (248., 294),  # 外部湿球温度
                               'zon_weaSta_reaWeaNTot_y': (0., 1.),  # 天空覆盖
                               'nTot': (0., 1.),  # 天空覆盖预测
                               'HDirNor': (0., 982.62),  # 直接辐射量预测
                               'HGloHor': (0., 1027.25),  # 全球水平太阳辐射预测
                               'relHum': (0., 1),  # 相对湿度预测
                               'TDryBul': (248., 310),  # 外部干球温度预测
                               'TWetBul': (248., 294),  # 外部湿球温度预测
                               'PriceElectricPowerDynamic': (0.04440, 0.13814),  # 电价
                                'UpperSetp[1]': (280., 310.),
                               'LowerSetp[1]': (280., 310.),
                               },

                 reward=['reward'],
                 max_episode_length=168 * 3600,
                 random_start_time=True,
                 excluding_periods=None,
                 regressive_period=None,
                 predictive_period=24 * 3600,
                 start_time=0,
                 warmup_period=7 * 24 * 3600,
                 scenario={'electricity_price': 'dynamic'},
                 step_period=3600,
                 render_episodes=False,
                 log_dir=os.getcwd()):
        '''
        Parameters
        ----------
        url: string
            Rest API url for communication with the BOPTEST interface
        actions: list
            List of strings indicating the action space. The bounds of
            each variable from the action space the are retrieved from
            the overwrite block attributes of the BOPTEST test case
        observations: dictionary
            Dictionary mapping observation keys to a tuple with the lower
            and upper bound of each observation. Observation keys must
            belong either to the set of measurements or to the set of
            forecasting variables of the BOPTEST test case. Contrary to
            the actions, the expected minimum and maximum values of the
            measurement and forecasting variables are not provided from
            the BOPTEST framework, although they are still relevant here
            e.g. for normalization or discretization. Therefore, these
            bounds need to be provided by the user.
            If `time` is included as an observation, the time in seconds
            will be passed to the agent. This is the remainder time from
            the beginning of the episode and for periods of the length
            specified in the upper bound of the time feature.
        reward: list
            List with string indicating the reward column name in a replay
            buffer of data in case the algorithm is going to use pretraining
        max_episode_length: integer
            Maximum duration of each episode in seconds
        random_start_time: boolean
            Set to True if desired to use a random start time for each episode
        excluding_periods: list of tuples
            List where each element is a tuple indicating the start and
            end time of the periods that should not overlap with any
            episode used for training. Example:
            excluding_periods = [(31*24*3600,  31*24*3600+14*24*3600),
                                (304*24*3600, 304*24*3600+14*24*3600)]
            This is only used when `random_start_time=True`
        regressive_period: integer, default is None
            Number of seconds for the regressive horizon. The observations
            will be extended for each of the measurement variables indicated
            in the `observations` dictionary argument. Specifically, a number
            of `int(self.regressive_period/self.step_period)` observations per
            measurement variable will be included in the observation space.
            Each of these observations correspond to the past observation
            of the measurement variable `j` steps ago. This is used in partially
            observable MDPs to compensate for the hidden states.
            Note that it is NOT allowed to use `regressive_period=0` since that
            would represent a case where you want to include a measurement at
            the current time in the observation space, which is directly done
            when adding such measurement to the `observations` argument.
        predictive_period: integer, default is None
            Number of seconds for the prediction horizon. The observations
            will be extended for each of the predictive variables indicated
            in the `observations` dictionary argument. Specifically, a number
            of `int(self.predictive_period/self.step_period)` observations per
            predictive variable will be included in the observation space.
            Each of these observations correspond to the foresighted
            variable `i` steps ahead from the actual observation time.
            Note that it's allowed to use `predictive_period=0` when the
            intention is to retrieve boundary condition data at the actual
            observation time, useful e.g. for temperature setpoints or
            ambient temperature.
        start_time: integer
            Initial fixed episode time in seconds from beginning of the
            year for each episode. Use in combination with
            `random_start_time=False`
        warmup_period: integer
            Desired simulation period to initialize each episode
        scenario: dictionary
            Defines the BOPTEST scenario. Can be `constant`, `dynamic` or
            `highly_dynamic`
        step_period: integer
            Sampling time in seconds
        render_episodes: boolean
            True to render every episode
        log_dir: string
            Directory to store results like plots or KPIs

        '''

        super(BoptestGymEnv, self).__init__()

        self.url = url
        self.actions = actions
        self.observations = list(observations.keys())
        self.max_episode_length = max_episode_length
        self.random_start_time = random_start_time
        self.excluding_periods = excluding_periods
        self.start_time = start_time
        self.warmup_period = warmup_period
        self.reward = reward
        self.predictive_period = predictive_period
        self.regressive_period = regressive_period
        self.step_period = step_period
        self.scenario = scenario
        self.render_episodes = render_episodes
        self.log_dir = log_dir
        self.min_max_info = min_max_info

        self.prediction_list = ['nTot',  # 天空覆盖预测
                                'HDirNor',  # 直接辐射量预测
                                'HGloHor',  # 全球水平太阳辐射预测
                                'relHum',  # 相对湿度预测
                                'TDryBul',  # 外部干球温度预测
                                'TWetBul', ]  # 外部湿球温度预测

        # Avoid requesting data before the beginning of the year
        if self.regressive_period is not None:
            self.bgn_year_margin = self.regressive_period
        else:
            self.bgn_year_margin = 0
        # Avoid surpassing the end of the year during an episode
        self.end_year_margin = self.max_episode_length

        # =============================================================
        # Get test information
        # =============================================================
        # Test case name
        self.name = requests.get('{0}/name'.format(url)).json()['payload']
        # Measurements available
        self.all_measurement_vars = requests.get('{0}/measurements'.format(url)).json()['payload']
        # Predictive variables available
        self.all_predictive_vars = requests.get('{0}/forecast_points'.format(url)).json()['payload']
        # Inputs available
        self.all_input_vars = requests.get('{0}/inputs'.format(url)).json()['payload']
        # Default simulation step
        self.step_def = requests.get('{0}/step'.format(url)).json()['payload']
        # Default scenario
        self.scenario_def = requests.get('{0}/scenario'.format(url)).json()['payload']

        # =============================================================
        # Define observation space
        # =============================================================
        # Assert size of tuples associated to observations
        for obs in self.observations:
            if len(observations[obs]) != 2:
                raise ValueError( \
                    'Values of the observation dictionary must be tuples ' \
                    'of dimension 2 indicating the expected lower and ' \
                    'upper bounds of each variable. ' \
                    'Variable "{}" does not follow this format. '.format(obs))

        # Assert that observations belong either to measurements or to predictive variables
        for obs in self.observations:
            if not (obs == 'time' or obs in self.all_measurement_vars.keys() or obs in self.all_predictive_vars.keys()):
                raise ReferenceError( \
                    '"{0}" does not belong to neither the set of ' \
                    'test case measurements nor to the set of ' \
                    'forecasted variables. \n' \
                    'Set of measurements: \n{1}\n' \
                    'Set of forecasting variables: \n{2}'.format(obs,
                                                                 list(self.all_measurement_vars.keys()),
                                                                 list(self.all_predictive_vars.keys())))

        # observations = measurements + predictions
        self.measurement_vars = [obs for obs in self.observations if (obs in self.all_measurement_vars)]

        # Initialize observations and bounds
        self.observations = []
        self.lower_obs_bounds = []
        self.upper_obs_bounds = []

        # Check for time in observations
        if 'time' in list(observations.keys()):
            self.observations.extend(['time'])
            self.lower_obs_bounds.extend([observations['time'][0]])
            self.upper_obs_bounds.extend([observations['time'][1]])

        # Define lower and upper bounds for observations. Always start observation space by measurements
        self.observations.extend(self.measurement_vars)
        self.lower_obs_bounds.extend([observations[obs][0] for obs in self.measurement_vars])
        self.upper_obs_bounds.extend([observations[obs][1] for obs in self.measurement_vars])

        # Check if agent uses regressive states and extend observations with these
        self.is_regressive = False
        if self.regressive_period is not None:
            self.is_regressive = True
            # Do a sanity check
            if self.regressive_period == 0 or self.regressive_period < 0:
                raise ValueError( \
                    'The regressive_period cannot be 0 or negative. ' \
                    'If you just want to add a measurement variabe to the ' \
                    'set of observations it is enough to add it to the ' \
                    'observations argument. ')
            self.regressive_vars = self.measurement_vars

            # Number of discrete regressive steps.
            # If regressive_period=3600, and step_period=900
            # then we have 4 regressive steps:
            # regr_1, regr_2, regr_3, regr_4 (actual not taken here)
            # regr_4 is the time step furthest away in the past
            self.regr_n = int(self.regressive_period / self.step_period)

            # Extend observations to have one observation per regressive step
            for obs in self.regressive_vars:
                obs_list = [obs + '_regr_{}'.format(int(i * self.step_period)) for i in range(1, self.regr_n + 1)]
                obs_lbou = [observations[obs][0]] * len(obs_list)
                obs_ubou = [observations[obs][1]] * len(obs_list)
                self.observations.extend(obs_list)
                self.lower_obs_bounds.extend(obs_lbou)
                self.upper_obs_bounds.extend(obs_ubou)

        # Check if agent uses predictions in state and parse predictive variables
        self.is_predictive = False
        self.predictive_vars = []
        if any([obs in self.all_predictive_vars for obs in observations]):
            self.is_predictive = True

            # Do a sanity check
            if self.predictive_period < 0:
                raise ValueError( \
                    'The predictive_period cannot be negative. ' \
                    'Set the predictive_period to be 0 or higher than 0 ')

            # Parse predictive vars
            self.predictive_vars = [obs for obs in observations if \
                                    (obs in self.all_predictive_vars and obs != 'time')]

            # Number of discrete predictive steps. If predictive_period=0,
            # then only 1 step is taken: the actual time step.
            # If predictive_period=3600, and step_period=900
            # then we have 5 predictive steps:
            # pred_0, pred_1, pred_2, pred_3, pred_4 (actual taken here)
            # pred_4 is the time step furthest away in the future
            self.pred_n = int(self.predictive_period / self.step_period)  # 这里修改了 去掉了+1

            # Extend observations to have one observation per predictive step
            for obs in self.predictive_vars:
                obs_list = [obs + '_pred_{}'.format(int(i * self.step_period)) for i in range(self.pred_n)]
                obs_lbou = [observations[obs][0]] * len(obs_list)
                obs_ubou = [observations[obs][1]] * len(obs_list)
                self.observations.extend(obs_list)
                self.lower_obs_bounds.extend(obs_lbou)
                self.upper_obs_bounds.extend(obs_ubou)

            # If predictive, the margin should be extended
            self.end_year_margin = self.max_episode_length + self.predictive_period

        # Define gym observation space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(
        len(self.predictive_vars) * (self.pred_n - 1) + len(self.measurement_vars) + 4 + 2 - 3 + 1,), dtype=np.float32)

        # =============================================================
        # Define action space
        # =============================================================
        # Assert that actions belong to the inputs in the emulator model
        for act in self.actions:
            if not (act in self.all_input_vars.keys()):
                raise ReferenceError( \
                    '"{0}" does not belong to the set of inputs to this ' \
                    'emulator model. \n' \
                    'Set of inputs: \n{1}\n'.format(act, list(self.all_input_vars.keys())))

        # Parse minimum and maximum values for actions
        self.lower_act_bounds = []
        self.upper_act_bounds = []
        for act in self.actions:
            self.lower_act_bounds.append(self.all_input_vars[act]['Minimum'])
            self.upper_act_bounds.append(self.all_input_vars[act]['Maximum'])

        # Define gym action space
        self.action_space = spaces.Box(low=np.array(self.lower_act_bounds),
                                       high=np.array(self.upper_act_bounds),
                                       dtype=np.float32)

        if self.render_episodes:
            plt.ion()
            self.fig = plt.gcf()

    def reset(self):
        '''
        Method to reset the environment. The associated building model is
        initialized by running the baseline controller for a
        `self.warmup_period` of time right before `self.start_time`.
        If `self.random_start_time` is True, a random time is assigned
        to `self.start_time` such that there are not episodes that overlap
        with the indicated `self.excluding_periods`. This is useful to
        define testing periods that should not use data from training.

        Returns
        -------
        observations: numpy array
            Reformatted observations that include measurements and
            predictions (if any) at the end of the initialization.

        '''

        def find_start_time():
            '''Recursive method to find a random start time out of
            `excluding_periods`. An episode and an excluding_period that
            are just touching each other are not considered as being
            overlapped.

            '''
            start_time = random.randint(0 + self.bgn_year_margin,
                                        3.1536e+7 - self.end_year_margin)
            episode = (start_time, start_time + self.max_episode_length)
            if self.excluding_periods is not None:
                for period in self.excluding_periods:
                    if episode[0] < period[1] and period[0] < episode[1]:
                        # There is overlapping between episode and this period
                        # Try to find a good starting time again
                        start_time = find_start_time()
            # This point is reached only when a good starting point is found
            return start_time

        # Assign random start_time if it is None
        if self.random_start_time:
            self.start_time = find_start_time()

        # Initialize the building simulation
        res = requests.put('{0}/initialize'.format(self.url),
                           data={'start_time': self.start_time,
                                 'warmup_period': self.warmup_period}).json()['payload']

        # Set simulation step
        requests.put('{0}/step'.format(self.url), data={'step': self.step_period})

        # Set BOPTEST scenario
        requests.put('{0}/scenario'.format(self.url), data=self.scenario)

        # Set forecasting parameters if predictive
        # if self.is_predictive:
        #     forecast_parameters = {'horizon':self.predictive_period, 'interval':self.step_period}
        #     requests.put('{0}/forecast_parameters'.format(self.url),
        #                  data=forecast_parameters)

        # Initialize objective integrand
        self.objective_integrand = 0.

        # Get observations at the end of the initialization period
        observations = self.get_observations(res)

        self.episode_rewards = []

        return observations

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

        return [seed]

    def step(self, action):
        '''
        Advance the simulation one time step

        Parameters
        ----------
        action: list
            List of actions computed by the agent to be implemented
            in this step

        Returns
        -------
        observations: numpy array
            Observations at the end of this time step
        reward: float
            Reward for the state-action pair implemented
        done: boolean
            True if episode is finished after this step
        info: dictionary
            Additional information for this step

        '''

        # Initialize inputs to send through BOPTEST Rest API
        u = {}

        # Assign values to inputs if any
        for i, act in enumerate(self.actions):
            # Assign value
            u[act] = action[i]

            # Indicate that the input is active
            u[act.replace('_u', '_activate')] = 1.

        # Advance a BOPTEST simulation
        res = requests.post('{0}/advance'.format(self.url), data=u).json()['payload']

        # Compute reward of this (state-action-state') tuple
        reward = self.compute_reward(res)
        self.episode_rewards.append(reward)

        # Define whether we've finished the episode
        done = self.compute_done(res, reward)

        # Optionally we can pass additional info, we are not using that for now
        info = {'indoor_temp': res['zon_reaTRooAir_y'] - 273.15}

        # Get observations at the end of this time step
        observations = self.get_observations(res)

        # Render episode if finished and requested
        if done and self.render_episodes:
            self.render()

        return observations, reward, done, info

    def render(self, mode='episodes'):
        '''
        Renders the process evolution

        Parameters
        ----------
        mode: string
            Mode to be used for the renderization

        '''
        if mode != 'episodes':
            raise NotImplementedError()
        else:
            plt.ion()
            self.fig = plt.gcf()
            self.fig.clear()
            plot_results(self, self.episode_rewards, log_dir=self.log_dir)

    def close(self):
        pass

    def compute_reward(self, res):
        '''
        Compute the reward of last state-action-state' tuple. The
        reward is implemented as the negated increase in the objective
        integrand function. In turn, this objective integrand function
        is calculated as the sum of the total operational cost plus
        the weighted discomfort.

        Returns
        -------
        Reward: float
            Reward of last state-action-state' tuple

        Notes
        -----
        This method is just a default method to compute reward. It can be
        overridden by defining a child from this class with
        this same method name, i.e. `compute_reward`. If a custom reward
        is defined, it is strongly recommended to derive it using the KPIs
        as returned from the BOPTEST framework, as it is done in this
        default `compute_reward` method. This ensures that all variables
        that may contribute to any KPI are properly accounted and
        integrated.

        '''

        # Define a relative weight for the discomfort
        w = 16

        # Compute BOPTEST core kpis
        kpis = requests.get('{0}/kpi'.format(self.url)).json()['payload']

        # Calculate objective integrand function at this point
        objective_integrand = w * kpis['cost_tot'] + kpis['tdis_tot']

        # Compute reward
        reward = -(objective_integrand - self.objective_integrand)

        self.objective_integrand = objective_integrand

        # reward = 1 / (np.abs(res['zon_reaTRooAir_y']-273.15 - 20) + 1)

        # error = np.abs(res['zon_reaTRooAir_y'] - 273.15)
        #
        # u = 20  # 均值μ
        # # u01 = -4
        # sig = math.sqrt(4)  # 标准差δ
        # deta = 5 * 1.6
        # reward = (deta * np.exp(-(error - u) ** 2 / (2 * sig ** 2)) / (math.sqrt(2 * math.pi) * sig)) - 0.6

        return reward

    def compute_done(self, res, reward=None):
        '''
        Compute whether the episode is finished or not. By default, a
        maximum episode length is defined and the episode will be finished
        only when the time exceeds this maximum episode length.

        Returns
        -------
        done: boolean
            Boolean indicating whether the episode is done or not.

        Notes
        -----
        This method is just a default method to determine if an episode is
        finished or not. It can be overridden by defining a child from
        this class with this same method name, i.e. `compute_done`. Notice
        that the reward for each step is passed here to enable the user to
        access this reward as it may be handy when defining a custom
        method for `compute_done`.

        '''

        done = res['time'] >= self.start_time + self.max_episode_length

        return done

    def get_observations(self, res):
        '''
        Get the observations, i.e. the conjunction of measurements,
        regressive and predictive variables if any. Also transforms
        the output to have the right format.

        Parameters
        ----------
        res: dictionary
            Dictionary mapping simulation variables and their value at the
            end of the last time step.

        Returns
        -------
        observations: numpy array
            Reformatted observations that include measurements and
            predictions (if any) at the end of last step.

        '''

        # Initialize observations
        observations = []

        # 添加相应的时间信息到observations中，共计4个
        start_of_year = datetime.datetime(year=2023, month=1, day=1)

        delta = datetime.timedelta(seconds=res['time'])

        new_date_and_time = start_of_year + delta

        # observations.append(new_date_and_time.month / 12)
        #
        # observations.append(new_date_and_time.hour / 23)
        #
        # observations.append(new_date_and_time.day / 31)

        observations.append((res['time'] - self.start_time) / self.max_episode_length)

        # Get measurements at the end of the simulation step
        for obs in self.measurement_vars:
            obs_min = self.min_max_info[obs][0]

            obs_max = self.min_max_info[obs][1]

            normalized_obs = (res[obs] - obs_min) / (obs_max - obs_min)

            observations.append(normalized_obs)

        # Get regressions if this is a regressive agent
        if self.is_regressive:
            regr_index = res['time'] - self.step_period * np.arange(1, self.regr_n + 1)
            for var in self.regressive_vars:
                res_var = requests.put('{0}/results'.format(self.url),
                                       data={'point_name': var,
                                             'start_time': regr_index[-1],
                                             'final_time': regr_index[0]}).json()['payload']
                # fill_value='extrapolate' is needed for the very few cases when
                # res_var['time'] is not returned to be exactly between
                # regr_index[-1] and regr_index[0] but shorter. In these cases
                # we extrapolate linearly to reach the desired value at the extreme
                # of the regression period.
                f = interpolate.interp1d(res_var['time'],
                                         res_var[var], kind='linear', fill_value='extrapolate')
                res_var_reindexed = f(regr_index)
                observations.extend(list(res_var_reindexed))

        # Get predictions if this is a predictive agent
        if self.is_predictive:

            predictions = requests.put('{0}/forecast'.format(self.url),
                                       data={'point_names': self.predictive_vars, 'horizon': self.predictive_period,
                                             'interval': self.step_period}).json()['payload']
            for var in self.predictive_vars:

                predict_min = self.min_max_info[var][0]

                predict_max = self.min_max_info[var][1]

                if var == 'PriceElectricPowerDynamic' or var == 'UpperSetp[1]' or var == 'LowerSetp[1]':
                    for i in range(self.pred_n):
                        normalized_pred = (predictions[var][i] - predict_min) / (predict_max - predict_min)
                        observations.append(normalized_pred)

                else:

                    for i in range(1, self.pred_n):
                        normalized_pred = (predictions[var][i] - predict_min) / (predict_max - predict_min)

                        observations.append(normalized_pred)

        # Reformat observations
        observations = np.array(observations).astype(np.float32)

        return observations

    def get_kpis(self):
        '''Auxiliary method to get the so-colled core KPIs as computed in
        the BOPTEST framework. This is handy when evaluating performance
        of an agent in this environment.

        '''

        # Compute BOPTEST core kpis
        kpis = requests.get('{0}/kpi'.format(self.url)).json()['payload']

        return kpis

    def reformat_expert_traj(self, file_path='data.csv'):
        '''
        Reformats expert trajectory from a csv file to the npz format
        required by Stable Baselines algorithms to be pre-trained.

        Parameters
        ----------
        file_path: string
            path to csv file containing data

        Returns
        -------
        numpy_dict: numpy dictionary
            Numpy dictionary with the reformatted data

        Notes
        -----
        The resulting reformatted data considers only one episode from
        a long trajectory (a long time series). No recurrent policies
        supported (mask and state not defined).

        '''

        # We consider only one episode of index 0 that is never done
        n_episodes = 1
        ep_idx = 0
        done = False

        # Initialize data in the episode
        actions = []
        observations = []
        rewards = []
        episode_returns = np.zeros((n_episodes,))
        episode_starts = []

        # Initialize the only episode that we use
        episode_starts.append(True)
        reward_sum = 0.0

        df = pd.read_csv(file_path)
        for row in df.index:
            # Retrieve step information from csv
            obs = df.loc[row, self.observations]
            action = df.loc[row, self.actions]
            reward = df.loc[row, self.reward]

            if obs.hasnans or action.hasnans or reward.hasnans:
                raise ValueError('Nans found in row {}'.format(row))

            # Append to data
            observations.append(np.array(obs))
            actions.append(np.array(action))
            rewards.append(np.array(reward))
            episode_starts.append(np.array(done))

            reward_sum += reward

        # This is hard coded as we only support one episode so far but
        # here we could implement some functionality for creating different
        # episodes from csv data
        done = True
        if done:
            episode_returns[ep_idx] = reward_sum
            reward_sum = 0.0

        if isinstance(self.observation_space, spaces.Box):
            observations = np.concatenate(observations).reshape((-1,) + self.observation_space.shape)
        elif isinstance(self.observation_space, spaces.Discrete):
            observations = np.array(observations).reshape((-1, 1))

        if isinstance(self.action_space, spaces.Box):
            actions = np.concatenate(actions).reshape((-1,) + self.action_space.shape)
        elif isinstance(self.action_space, spaces.Discrete):
            actions = np.array(actions).reshape((-1, 1))

        rewards = np.array(rewards)
        episode_starts = np.array(episode_starts[:-1])

        assert len(observations) == len(actions)

        numpy_dict = {
            'actions': actions,
            'obs': observations,
            'rewards': rewards,
            'episode_returns': episode_returns,
            'episode_starts': episode_starts
        }  # type: Dict[str, np.ndarray]

        for key, val in numpy_dict.items():
            print(key, val.shape)

        np.savez(file_path.split('.')[-2], **numpy_dict)

        return numpy_dict


class CustomLSTM(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        self.model = nn.GRU(input_size=9 + 2, hidden_size=32, num_layers=1, bidirectional=False,
                             batch_first=True)


        # Compute shape by doing one forward pass
        # with torch.no_grad():
        #     lstm_output, (h_n, c_n) = self.model(
        #         torch.as_tensor(observation_space.sample()[None]).float())a

        n_flatten = 24 * 32

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations):
        # 218个维度，第一个特征是在episode中的位置，第二个是室内温度，然后是6个不含电价的测量值,然后是6个值的23小时预测按顺序，最后是24小时的电价, 24小时温度上限，24小时温度下限

        batch_size = observations.shape[0]

        current_input = observations[:, :2]

        lstm_pre = observations[:, 2:]

            # 创建一个零张量来存储结果
        lstm_input = torch.zeros(batch_size, 24, 9).to(observations.device)

            # 把前6个值放在第一个时间步
        lstm_input[:, 0, :6] = lstm_pre[:, :6]

            # 把后138个值放入相应的位置
        for i in range(6):
            lstm_input[:, 1:, i] = lstm_pre[:, 6 + i * 23: 6 + i * 23 + 23].view(batch_size, 23)

            # 把第七个变量的值放入相应的位置
        lstm_input[:, :, 6] = lstm_pre[:, 144:144 + 24].view(batch_size, 24)

        lstm_input[:, :, 7] = lstm_pre[:, 144 + 24:144 + 48].view(batch_size, 24)

        lstm_input[:, :, 8] = lstm_pre[:, 144 + 48: ].view(batch_size, 24)

        current_input = current_input.unsqueeze(1).expand(batch_size, 24, -1)

        lstm_input = torch.cat([lstm_input, current_input], dim=-1)

        lstm_output, _ = self.model(lstm_input)

            # current_output = self.current_model(current_input)

        lstm_output = lstm_output.reshape(batch_size, -1)
            #
            # lstm_output = torch.cat([lstm_output, current_output], dim=-1)

        return self.linear(lstm_output)

policy_kwargs = dict(
        features_extractor_class=CustomLSTM,
        features_extractor_kwargs=dict(features_dim=256),
    )

if __name__ == "__main__":
    # 首先设定随机种子
    #
    random_seed = 1993

    # Instantiate the env
    env = BoptestGymEnv()

    env.seed(random_seed)

    env.action_space.seed(random_seed)


    def seed_everything(seed):
        torch.manual_seed(seed)  # Current CPU
        torch.cuda.manual_seed(seed)  # Current GPU
        np.random.seed(seed)  # Numpy module
        random.seed(seed)  # Python random module
        torch.backends.cudnn.benchmark = False  # Close optimization
        torch.backends.cudnn.deterministic = True  # Close optimization
        torch.cuda.manual_seed_all(seed)  # All GPU (Optional)


    #
    seed_everything(random_seed)

    # Check the environment
    # check_env(env)
    # obs = env.reset()

    #
    # done = False
    # env.reset()
    # while not done:
    #   state, reward, done, _ = env.step(env.action_space.sample())
    #   print(reward)
    #   #print(f'Checking if the state is part of the observation space: {env.observation_space.contains(state)}')
    #
    # env.close()

    # env.render()
    # print('Observation space: {}'.format(env.observation_space))
    # print('Action space: {}'.format(env.action_space))

    model = SAC("MlpPolicy", env, verbose=66, seed=random_seed, learning_rate=0.0001,
                tensorboard_log='./td3_tensorboard/', policy_kwargs=policy_kwargs, batch_size=512)

    model.learn(total_timesteps=1000 * 168)

    name = 'SAC_上下限'

    model.save(name)

    # 这里开始测试一下
    # model = TD3.load("TD366.zip")
    #
    # obs = env.reset()
    # done = False
    #
    # df = pd.DataFrame(columns=['indoor_temp'])
    #
    # temp_list = []
    #
    # while not done:
    #     action, _states = model.predict(obs, deterministic=True)
    #
    #     obs, reward, done, info = env.step(action)
    #
    #     temp_list.append(info['indoor_temp'])
    #
    #
    # df['indoor_temp'] = temp_list
    #
    # df.to_csv('result.csv')







