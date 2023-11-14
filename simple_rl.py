import time                 # to measure the computation time
import gym
from gym import spaces, core
import numpy as np
import random
import pandas as pd
import math
import os

from stable_baselines3 import TD3
from stable_baselines3 import PPO
from stable_baselines3 import DDPG
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
import stable_baselines3

import matplotlib.pyplot as plt
import seaborn as sns

from stable_baselines3.common.callbacks import EvalCallback


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True









# 如果把168小时的工作日和休息日全部求和，再平均值，作为一个状态


class Takasago_ENV(gym.Env):
  """A building energy system operational optimization for OpenAI gym and takasago"""

  def __init__(self):

    super(Takasago_ENV, self).__init__()
    self.data = pd.read_csv('train.csv')
    self.data = self.data.rename(columns=lambda x: x.strip())

    self.action_space = spaces.Box(
      low=np.array([0, 0.25, -1]), high=np.array([1, 1, 1]), dtype=np.float32)
    self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)

    self.Max_temp = self.data.max()['temperature']
    self.Max_pv = self.data.max()['PV_output']
    self.Max_solar = self.data.max()['Solar']
    self.Max_hour = 23
    self.Max_power = self.data.max()['Whole']
    self.history_length = 24
    self.optimize_length = 168
    self.battery_change = 200

    # 数据标记
    self.time = 0
    self.t = 0

    # 电机动作差
    self.bio_action = 1

  def _next_observation(self):

    history_frame = np.array([
      self.data.loc[self.current_step, 'holiday'],
      self.data.loc[self.current_step, 'hour'] / self.Max_hour,
      self.data.loc[self.current_step, 'Whole'] / self.Max_power,
      self.data.loc[self.current_step, 'PV_output'] / self.Max_pv,
      self.data.loc[self.current_step, 'Solar'] / self.Max_solar,
      # self.data.loc[self.original_step:self.original_step + self.optimize_length - 1, 'holiday']
      # self.data.loc[self.current_step, 'Temperature'] / self.Max_temp,
      # self.data.loc[self.current_step, 'Humidity'] / 100,
      #self.data.loc[self.current_step, 'day_of_year'] / 365,
      #self.data.loc[self.current_step, 'dayofweek'] / 7,
      #self.data.loc[self.current_step, 'month'] / 12,
      #self.data.loc[self.current_step, 'temperature'] / self.Max_temp,


    ])

    #future_array = np.array([self.data.loc[self.original_step:self.original_step + self.optimize_length - 1, 'holiday']]).squeeze()

    step_in_epo = (self.current_step - self.original_step) / self.optimize_length

    obs = np.append(history_frame, self.battery_state)
    #obs = np.append(obs, future_array)
    obs = np.append(obs, step_in_epo)

    return obs.astype(np.float32)

  def reset(self):
    # Reset the state of the environment to an initial state
    self.battery_state = np.random.randint(20, 80) * 0.01
    #self.battery_state = 0.5
    # Set the current step to a random point within the data frame
    self.current_step = random.randint(0, self.data.shape[0] - self.optimize_length - 1)
    #self.current_step = 4596

    self.original_step = self.current_step

    return self._next_observation()

  def reward(self, error, act, battery, battery_action):
    u = -10  # 均值μ
    # u01 = -4
    sig = math.sqrt(4)  # 标准差δ
    deta = 5 * 1.6
    reward_1 = (deta * np.exp(-(error - u) ** 2 / (2 * sig ** 2)) / (math.sqrt(2 * math.pi) * sig)) - 0.6

    reward_2 = -0.1 * np.abs(self.bio_action - act)

    # if error > 0:
    #     reward_4 = -1
    # elif error < - 100:
    #     reward_4 = -1
    # else:
    #     reward_4 = 0

    if battery <= 0.2 and battery_action < 0:
      # reward_3 = 0.25 * battery - 0.1
      reward_3 = -1.2

    elif battery > 0.8 and battery_action > 0:
      # reward_3 = (-battery + 0.6) * 0.25
      reward_3 = -1.2
    else:
      reward_3 = 0

    self.bio_action = act

    rewards = reward_1
    return rewards + reward_3

  def step(self, action):

    # 读取计算结果，计算reward
    if self.current_step - self.original_step > self.optimize_length - 2:
      done = True
    else:
      done = False

    current_battery = self.battery_state

    # 电池动作
    if (action[-1] * self.battery_change) / 4595 + self.battery_state < 0:
      action[-1] = - self.battery_state  # 改变动作，只能放这么多电
      reward_battery = -0.1  # 动作错误，一个负奖励
      self.battery_state = 0

    elif (action[-1] * self.battery_change) / 4595 + self.battery_state > 1:
      action[-1] = 1 - self.battery_state
      reward_battery = -0.1
      self.battery_state = 1

    else:
      reward_battery = 0
      self.battery_state = self.battery_state + (action[-1] * self.battery_change) / 4595

    # 动作归一化

    #action[0] = (action[0] - (-1)) / 2

    #action[1] = (action[1] - (-1)) / 2

    # 电机动作离散化

    # if 0 <= action[1] <= 0.33:
    #   action[1] = 0
    # elif 0.33 < action[1] <= 0.66:
    #   action[1] = 1
    # else:
    #   action[1] = 2

    pv_gen = action[0] * self.data.iloc[self.current_step]['PV_output']
    bio_gen = action[1] * 80

    error = bio_gen + pv_gen - self.data.iloc[self.current_step]['Whole'] - action[2] * self.battery_change

    # if error > 0:
    #     done = True
    # else:
    #     done = False

    reward = self.reward(error, action[1], self.battery_state, action[2])

    reward = reward
    #
    self.current_step += 1
    state = self._next_observation()

    return state, reward, done, {'step': self.current_step, 'error': error, 'pv_action': action[0],
                                 'bio_action': action[1], 'battery_action': action[2], 'battery_state': current_battery}


class Takasago_ENV_TEST(gym.Env):
  """A building energy system operational optimization for OpenAI gym and takasago"""

  def __init__(self):

    super(Takasago_ENV_TEST, self).__init__()
    self.data = pd.read_csv('test.csv')
    self.data_cal = pd.read_csv('train.csv')
    self.data = self.data.rename(columns=lambda x: x.strip())

    self.action_space = spaces.Box(
      low=np.array([0, 0.25, -1]), high=np.array([1, 1, 1]), dtype=np.float32)
    self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)

    self.Max_temp = self.data_cal.max()['temperature']
    self.Max_pv = self.data_cal.max()['PV_output']
    self.Max_solar = self.data_cal.max()['Solar']
    self.Max_hour = 23
    self.Max_power = self.data_cal.max()['Whole']
    self.history_length = 24
    self.optimize_length = 168
    self.battery_change = 200

    # 数据标记
    self.time = 0
    self.t = 0

    # 电机动作差


  def _next_observation(self):

    history_frame = np.array([
      self.data.loc[self.current_step, 'holiday'],
      self.data.loc[self.current_step, 'hour'] / self.Max_hour,
      self.data.loc[self.current_step, 'Whole'] / self.Max_power,
      self.data.loc[self.current_step, 'PV_output'] / self.Max_pv,
      self.data.loc[self.current_step, 'Solar'] / self.Max_solar,
      # self.data.loc[self.original_step:self.original_step + self.optimize_length - 1, 'holiday']
      # self.data.loc[self.current_step, 'Temperature'] / self.Max_temp,
      # self.data.loc[self.current_step, 'Humidity'] / 100,
      #self.data.loc[self.current_step, 'day_of_year'] / 365,
      #self.data.loc[self.current_step, 'dayofweek'] / 7,
      #self.data.loc[self.current_step, 'month'] / 12,
      #self.data.loc[self.current_step, 'temperature'] / self.Max_temp,


    ])

    future_array = np.array([self.data.loc[self.original_step:self.original_step + self.optimize_length - 1, 'holiday']]).squeeze() * 0.1

    step_in_epo = (self.current_step - self.original_step) / self.optimize_length

    obs = np.append(history_frame, self.battery_state)
    #obs = np.append(obs, future_array)
    obs = np.append(obs, step_in_epo)

    return obs.astype(np.float32)

  def reset(self):
    # Reset the state of the environment to an initial state
    self.battery_state = np.random.randint(20, 80) * 0.01
    #self.battery_state = 0.2
    # Set the current step to a random point within the data frame
    self.current_step = random.randint(0, self.data.shape[0] - self.optimize_length - 1)
    #self.current_step = 0

    self.original_step = self.current_step

    return self._next_observation()

  def reward(self, error, act, battery, battery_action):
    u = -10  # 均值μ
    # u01 = -4
    sig = math.sqrt(4)  # 标准差δ
    deta = 5 * 1.6
    reward_1 = (deta * np.exp(-(error - u) ** 2 / (2 * sig ** 2)) / (math.sqrt(2 * math.pi) * sig)) - 0.6

    #reward_2 = -0.1 * np.abs(self.bio_action - act)

    # if error > 0:
    #     reward_4 = -1
    # elif error < - 100:
    #     reward_4 = -1
    # else:
    #     reward_4 = 0

    if battery <= 0.2 and battery_action < 0:
      # reward_3 = 0.25 * battery - 0.1
      reward_3 = -1.2

    elif battery > 0.8 and battery_action > 0:
      # reward_3 = (-battery + 0.6) * 0.25
      reward_3 = -1.2
    else:
      reward_3 = 0

    self.bio_action = act

    rewards = reward_1
    return rewards + reward_3

  def step(self, action):

    # 读取计算结果，计算reward
    if self.current_step - self.original_step > self.optimize_length - 2:
      done = True
    else:
      done = False

    current_battery = self.battery_state

    # 电池动作
    if (action[-1] * self.battery_change) / 4595 + self.battery_state < 0:
      action[-1] = - self.battery_state  # 改变动作，只能放这么多电
      reward_battery = -0.1  # 动作错误，一个负奖励
      self.battery_state = 0

    elif (action[-1] * self.battery_change) / 4595 + self.battery_state > 1:
      action[-1] = 1 - self.battery_state
      reward_battery = -0.1
      self.battery_state = 1

    else:
      reward_battery = 0
      self.battery_state = self.battery_state + (action[-1] * self.battery_change) / 4595

    # 动作归一化

    #action[0] = (action[0] - (-1)) / 2

    #action[1] = (action[1] - (-1)) / 2

    # 电机动作离散化

    # if 0 <= action[1] <= 0.33:
    #   action[1] = 0
    # elif 0.33 < action[1] <= 0.66:
    #   action[1] = 1
    # else:
    #   action[1] = 2

    pv_gen = action[0] * self.data.iloc[self.current_step]['PV_output']
    bio_gen = action[1] * 80

    error = bio_gen + pv_gen - self.data.iloc[self.current_step]['Whole'] - action[2] * self.battery_change

    # if error > 0:
    #     done = True
    # else:
    #     done = False

    reward = self.reward(error, action[1], self.battery_state, action[2])

    reward = reward
    #
    self.current_step += 1
    state = self._next_observation()

    return state, reward, done, {'step': self.current_step, 'error': error, 'pv_action': action[0],
                                 'bio_action': action[1], 'battery_action': action[2], 'battery_state': current_battery}






#
if __name__ == "__main__":
    train_env = Takasago_ENV()
    eval_env = Takasago_ENV_TEST()

    #action_noise = stable_baselines3.common.noise.NormalActionNoise(0, 0.05)


    eval_callback = EvalCallback(eval_env=eval_env, best_model_save_path='./logs_eval_122/',
                                 log_path='./logs_eval_122/', eval_freq=10 * 168,
                                 deterministic=True, render=False)
    # log_dir = "tmp/"
    # os.makedirs(log_dir, exist_ok=True)

    # Create and wrap the environment
    #env = Monitor(env, log_dir)

    #callback = SaveOnBestTrainingRewardCallback(check_freq=168, log_dir=log_dir)









    #policy_kwargs = dict(net_arch=dict(pi=[400, 300, 300], qf=[400, 300, 300]))
    
    model = DDPG("MlpPolicy", train_env, verbose=1, learning_rate=0.0002, tensorboard_log='./td3_tensorboard/')
    #model.learn(total_timesteps=20000 * 168, callback=eval_callback)
    #model.save("DDPG")
    #del model
    #print("model:",model)
    model = DDPG.load("DDPG")
    env = train_env


    # model = PPO.load("ppo_takasago")
    # 创建几个空列表获取结果
    PV_gen = []
    Power_demand = []
    batteray_usage = []
    bio_gen = []
    error = []
    battery_state = []
    pv_action = []
    reward_all = []

    obs = env.reset()
    done = False
    data = pd.read_csv('train.csv').rename(columns=lambda x: x.strip())
    df = pd.DataFrame(columns=['bio_gen', 'battery_usage', 'pv_gen', 'power', 'error', 'battery_state', 'pv_action', 'reward'])
    print('开始最终测试')
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        Power_demand.append(data.loc[info['step'] - 1, 'Whole'])
        PV_gen.append(info['pv_action'] * data.loc[info['step'] - 1, 'PV_output'])
        bio_gen.append(info['bio_action'] * 80)
        batteray_usage.append(- info['battery_action'] * env.battery_change)
        error.append(info['error'])
        battery_state.append(info['battery_state'])
        pv_action.append(info['pv_action'])
        reward_all.append(reward)
        if -20 < info['error'] < 0:
            #print('good')
            #print(info['error'])
            continue
        else:
            print('not good')
            print(info['error'])

    df['battery_usage'] = batteray_usage
    df['bio_gen'] = bio_gen

    df['pv_gen'] = PV_gen
    df['power'] = Power_demand
    df['error'] = error
    df['battery_state'] = battery_state
    df['pv_action'] = pv_action
    df['reward']  = reward_all

    df.to_csv('rl.csv')























  
  

