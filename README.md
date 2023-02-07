## Package Needed
numpy

pandas>=1.1.5

stockstats

pyfolio

matplotlib

gym>=0.17

stable-baselines3


## 训练

环境文件：*env.py*

默认参数文件：*config.py*

通过stable_baselines3训练单个模型：*train_single.py*

自行修改参数，可训练模型有DDPG、PPO、TD3、SAC、A2C

训练全部模型：*train_all.ipynb*

## 测试

run *backtest.ipynb*

## 自建算法

*attention_ppo文件夹*：用于存放引入attention机制后PPO已训练模型

*ppo文件夹*：存放自建PPO的已训练模型

*PPO.py*: 自建的teamPPO算法

*PPO_train.ipynb*：用于训练自建PPO以及AttentionPPO

*Attention_PPO.py*：引入attention机制的PPO

## 其余文件

*backtest文件夹*：用于存储不同算法下的账户与动作信息

*data_file文件夹*：存储train set 以及 test set

*train_file文件夹*：用于存储训练好的模型

