## training

Train a PPO agent:

>python main.py --algo=ppo

(We assume the opponent is not moving at all.)


# Evaluating

>python main.py --load_model --algo=ppo --load_run=1 --load_episode=900
>
>python main.py --load_model --algo=ppo --load_run=2 --load_episode=900
>
>python main.py --load_model --algo=ppo --load_run=3 --load_episode=900
>
>python main.py --load_model --algo=ppo --load_run=4 --load_episode=1500



