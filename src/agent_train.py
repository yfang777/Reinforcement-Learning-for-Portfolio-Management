import numpy as np
import pandas as pd
import copy
import random
import os
import click

from FinRL.src.env import StockLearningEnv
import FinRL.src.config as config

from stable_baselines3 import DDPG
from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3 import TD3
from stable_baselines3 import SAC

from PPO_improve import PPO_improve
from PPO import PPO


def load_model(model_name, e_train_gym):
	if model_name == "A2C":
		model= A2C(policy='MlpPolicy', env=e_train_gym, **config.A2C_PARAMS)
		# model = DDPG(policy='MlpPolicy', env=e_train_gym, **config.DDPG_PARAMS)
		# model = PPO(policy='MlpPolicy', env=e_train_gym, **config.PPO_PARAMS)
		# model = SAC(policy='MlpPolicy', env=e_train_gym, **config.SAC_PARAMS)
		# model = TD3(policy='MlpPolicy', env=e_train_gym, **config.TD3_PARAMS)
	elif model_name == "PPO_improve":
		model = PPO_improve(**config.kwargs)
	else:
		raise NotImplementedError()
	return model

@click.option()
@click.command("--model_type", default="PPO_improve")
@click.command("--max_train_steps", default=50000)
@click.command("--update_interval", default=100, help="Model update episode")
@click.command("--save_interval", default=50000)
def agent_train(
	model_type,
	max_train_steps,
	update_interval,
	save_interval
):

	train_df = pd.read_csv('./data_file/train.csv')
	e_train_gym = StockLearningEnv(df=train_df, **config.ENV_TRAIN_PARAMS)

	model = load_model(model_name=model_type , e_train_gym=e_train_gym)
	
	traj_lenth = 0
	total_steps = 0

	while total_steps < max_train_steps:
		s, done, steps, ep_r = e_train_gym.reset(), False, 0, 0

	'''Interact & trian'''
	while not done:
		traj_lenth += 1
	# print('s:', s)
	s = np.array(s)
	# print(np.shape(s))
	a, logprob_a = model.select_action(s)

	s_prime, r, done, info = e_train_gym.step(a)

	'''distinguish done between dead|win(dw) and reach env._max_episode_steps(rmax); done = dead|win|rmax'''
	'''dw for TD_target and Adv; done for GAE'''
	if done :
		dw = True
	else:
		dw = False

	model.put_data((s, a, r, s_prime, logprob_a, done, dw))
	s = s_prime
	ep_r += r

	'''update if its time'''

	if traj_lenth % update_interval == 0:
		model.train()
		traj_lenth = 0

	total_steps += 1

	'''save model'''
	if total_steps % save_interval==0:
		model.save(total_steps)