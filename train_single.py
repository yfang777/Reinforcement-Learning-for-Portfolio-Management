import pandas as pd
import os

from stable_baselines3 import TD3
from stable_baselines3 import SAC

from env import StockLearningEnv
import config

df_train = pd.read_csv('data_file/train.csv')
e_train_gym = StockLearningEnv(df=df_train, **config.ENV_TRAIN_PARAMS)

if __name__ == '__main__':
    episode = 50000
    # 只对PPO做展示
    model = SAC(policy='MlpPolicy', env=e_train_gym, **config.SAC_PARAMS)
    model.load(os.path.join('train_file', "{}.model".format('SAC' + str(150000))))

    model.learn(total_timesteps=episode)
    model.save(os.path.join('train_file', "{}.model".format('SAC' + str(200000))))

    model1 = TD3(policy='MlpPolicy', env=e_train_gym, **config.TD3_PARAMS)
    model1.load(os.path.join('train_file', "{}.model".format('TD3' + str(150000))))

    model1.learn(total_timesteps=episode)
    model1.save(os.path.join('train_file', "{}.model".format('TD3' + str(200000))))
