# Competition_Olympics-Curling
---
## Update 24.Feb:
We add more information to the output of the environment. For each simulation step, the environment takes action from both agents as input and return the following dict to each agent:

    {
    obs         : partial observation of current controlled agent, filled with -1 when is not your turn;
    team color  : color of controlled agents(purple for agent 0, green for agent 1);
    release     : indicator of whether the agent has crossed the release line (shown as red line), and whether the environment will consider the received actions
    game round  : 0 or 1, indicating current game round;
    throws left : how many numbers of throws left, including the opponent team, maximum 4 for each game round;
    score       : current score board, including opponent team, updated when each game round ends;
    controlled_
    player_index: 0 or 1,  representing agent purple or green respectively
    }

We also bring the cross line (red) closer to the center of the curling field, as we want the agent to know more about the field to help decision making before crossing the line.

我们在冰壶环境的输出中添加了关于比赛的额外信息，包括控制队伍颜色，是否已释放冰壶，游戏局数，剩余投掷冰壶数量（包含对手数据），比分（包含对手比分）以及玩家编号；同时我们将冰壶释放红线移至更靠近场地中心位置（向下），给予智能体更多场地信息。


## Update 16.Feb:

We add one more game round to the curling environment with serving order switched. So in round one, agent purple start first and agent green finish last; while in round two agent green start first and agent purple finish last.
我们在冰壶环境中额外添加了一局比赛，并互换双方发球顺序。如：第一局紫色方先发球，第二局绿色方先发球。

After each round, a team scores one game point for each of its own stones closer to the center than any stone of the opposite team and only one team can score in the end of each round. After two game rounds, total game point will be computed and the winner has the highest total score.
每一局游戏结束时将会进行当局游戏得分结算，一方每有一个冰壶比另一方所有冰壶更靠近圆心则得一分，每局比赛仅有一方得分；两局比赛结束后，将根据两局比赛的总得分决出胜负，得分高的一方为获胜方。



---
## Environment

<img src=https://github.com/jidiai/Competition_Olympics-Curling/blob/main/olympics_engine/assets/olympics%20curling.gif width=600>


Check details in Jidi Competition [RLChina2022智能体竞赛](http://www.jidiai.cn/compete_detail?compete=14)


### “奥林匹克 冰壶” :
<b>标签：</b>不完全观测；连续动作空间；连续状态空间

<b>环境简介：</b>智能体参加奥林匹克运动会。在这个系列的竞赛中， 两个智能体参加冰壶竞赛，目标是将球推至目标中心点处。

<b>环境规则:</b> 
1. 对战双方各控制四个有相同质量和半径的弹性小球智能体；
2. 双方智能体轮流向场地中央的目标点抛掷小球，每方智能体有四次抛掷的机会；
3. 四个回合结束后，所抛掷小球离目标点近的一方取得胜利；
4. 智能体可以互相碰撞，也可以碰撞墙壁；
5. 智能体的视野限定为自身朝向前方30*30的矩阵区域；
6. 当回合结束时环境结束。

<b>动作空间：</b>连续；两维。分别代表施加力量和转向角度。

<b>观测：</b>每一步环境返回一个30x30的二维矩阵，详情请见 */olympics_engine*文件夹 以及其他对局信息。

<b>奖励函数:</b> 距离目标点近的一方得100分，否则得0分。

<b>环境终止条件:</b> 当回合结束时环境结束。

<b>评测说明：</b>该环境属于零和游戏，在金榜的积分按照ELO进行匹配算法进行计算并排名。零和游戏在匹配对手或队友时，按照瑞士轮进行匹配。
平台验证和评测时，在单核CPU上运行用户代码（暂不支持GPU），限制用户每一步返回动作的时间不超过1s，内存不超过500M。

<b>报名方式：</b>访问“及第”平台（ www.jidiai.cn ），在“擂台”页面选择“RLChina 智能体挑战赛 - 壬寅年春赛季”即可报名参赛。RLCN 微信公众号后台回复“智能体竞赛”，可进入竞赛讨论群。

This is a POMDP simulated environment of 2D sports games where althletes are spheres and have continuous action space (torque and steering). The observation is a 30*30 array of agent's limited view range. We introduce collision and agent's fatigue such that no torque applies when running out of energy.

This is for now a beta version and we intend to add more sports scenario, stay tuned :)

---
## Dependency

>conda create -n olympics python=3.8.5

>conda activate olympics

>pip install -r requirements.txt

---

## Run a game

>python olympics_engine/main.py

---

## Train a baseline agent 

>python rl_trainer/main.py

You can also locally evaluate your trained model by executing:

>python evaluation_local.py --my_ai rl --opponent random --episode=50


## How to test submission

You can locally test your submission. At Jidi platform, we evaluate your submission as same as *run_log.py*

For example,

>python run_log.py --my_ai "rl" --opponent "random"

in which you are controlling agent 1 which is green.

## Ready to submit

1. Random policy --> *agents/random/submission.py*
2. RL policy --> *all files in agents/rl*

