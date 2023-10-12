import numpy as np
import pandas as pd
import os
import warnings
import click
warnings.filterwarnings("ignore")

import FinRL.src.config as config
from FinRL.src.env import StockLearningEnv
from FinRL.src.PPO_improve import Attention_PPO
from FinRL.src.PPO import PPO


@click.command()
@click.option("--step_checkpoint", default=50000)
def test(step_checkpoint):

    model = PPO(**config.kwargs)
    model_name = 'PPO_vanilla'
    # model = Attention_PPO(**kwargs)
    # model_name = 'teamPPO'
    model.load(step_checkpoint)

    df = pd.read_csv('./data_file/trade.csv')
    e_trade_gym = StockLearningEnv(df=df, **config.ENV_TRADE_PARAMS)
    test_env, test_obs = e_trade_gym.get_sb_env()
    test_env.reset()

    len_environment = len(e_trade_gym.df.index.unique())

    for i in range(len_environment):
        action = model.select_action(test_obs)
        test_obs, _, dones, _ = test_env.step(action)
        if i == len_environment -2:
            df_account = test_env.env_method(method_name="save_asset_memory")[0]
            df_action = test_env.env_method(method_name="save_action_memory")[0]
            print("回测完成!")

    
    df_action.to_csv(f'./backtest/{model_name}_{step_checkpoint}_action.csv')
    df_account.to_csv(f'./backtest/{model_name}_{step_checkpoint}_account.csv')