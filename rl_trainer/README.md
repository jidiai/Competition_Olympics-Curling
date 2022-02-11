## training

Train a PPO agent against a random opponent:

>python main.py --algo=ppo --opponent=random --controlled_player=1



# Evaluating

>python main.py --algo=ppo --opponent=random --opponent_load_episode=1500 --render --load_model --load_run=1 --load_episode=1500 --controlled_player=1
>
>python main.py --algo=ppo --opponent=run1 --opponent_load_episode=1500 --render --load_model --load_run=2 --load_episode=1500 --controlled_player=1





