# maddpg_for_rlcn_snakes

 ## Dependency

You need to create competition environment.

>conda create -n snake1v1 python=3.6

>conda activate snake1v1

>pip install -r requirements.txt

## How to train rl-agent

>python rl_trainer/main.py

## You can edit different parameters, for example

>python rl_trainer/main.py --algo "bicnet" --epsilon 0.8

## You can locally evaluation your model.

>python evaluation_local.py --my_ai rl --opponent random

