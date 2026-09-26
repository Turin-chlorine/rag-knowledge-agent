强化学习
强化学习（英语：Reinforcement learning，简称RL）是机器学习中的⼀个领域，强调如何基于
环境⽽⾏动，以取得最⼤化的预期利益[1]。强化学习是除了监督学习和⾮监督学习之外的第三种基
本的机器学习⽅法。与监督学习不同的是，强化学习不需要带标签的输⼊输出对，同时也⽆需对⾮
最优解的精确地纠正。其关注点在于寻找（对未知领域的）探索和（对已有知识的）利⽤的平
衡[2]，强化学习中的“探索-利⽤”的交换，在多臂赌博机问题和有限MDP中研究得最多。
其灵感来源于⼼理学中的⾏为主义理论，即有机体如何在环境给予的奖励或惩罚的刺激下，逐步形
成对刺激的预期，产⽣能获得最⼤利益的习惯性⾏为。这个⽅法具有普适性，因此在其他许多领域
都有研究，例如博弈论、控制论、运筹学、信息论、仿真优化、多智能体系统、群体智能、统计学
以及遗传算法。在运筹学和控制理论研究的语境下，强化学习被称作“近似动态规划”
（approximate dynamic programming，ADP）。在最优控制理论中也有研究这个问题，虽然⼤
部分的研究是关于最优解的存在和特性，并⾮是学习或者近似⽅⾯。在经济学和博弈论中，强化学
习被⽤来解释在有限理性的条件下如何出现平衡。
在强化学习问题中，智能体（agent）与环境的交互通常被抽象为⻢尔可夫决策过程（Markov
decision processes，MDP），因为很多强化学习算法在这种假设下才能使⽤动态规划的⽅法[3]。
传统的动态规划⽅法和强化学习算法的主要区别是，后者不需要关于MDP的知识，⽽且针对⽆法找
到确切⽅法的⼤规模MDP。[4]
对于时刻 下观测（observation） 与环境实际状态 的关系，MDP可以被分为：
完全可观测MDP，若
部分可观测MDP (POMDP)，若 [5]
对于Q-learning，其对应的策略 可被定义为：
其中 为温度参数，⽤于控制策略的随机程度。 趋于 0 时，策略趋于贪⼼策略（
）； 趋于正⽆穷时，策略趋于均匀随机。[6]
介绍
由于其通⽤性很强，强化学习已经在诸如博弈论、控制论、运筹学、信息论、仿真优化、多智能体
系统、群体智能和统计学等领域有了深⼊研究。在运筹学和控制⽂献中，强化学习被称为近似动态
规划或神经动态规划。强化学习所感兴趣的问题在最优控制（⼀种关注最优解的存在性、表⽰和求
解的理论，但较少涉及学习和近似）中也有所研究，尤其是环境的数学模型难以求得的时候。在经
济学和博弈论中，强化学习可能被⽤来解释在有限的理性（rationality）下如何达到平衡状态。
基本的强化学习被建模为⻢尔可夫决策过程：
1. 环境状态的集合 ;
2. 动作的集合 ;
3. 在状态之间转换的规则（转移概率矩阵） ；
4. 规定转换后“即时奖励”的规则（奖励函数） ；
5. 描述主体能够观察到什么的规则。
规则通常是带有随机性的。主体通常可以观察即时奖励和最
后⼀次转换。在许多模型中，主体被假设为可以观察现有的
环 境 状 态 ， 这 种 情 况 称 为 “ 完 全 可 观 测 ”（ full
observability），反之则称为“部分可观测”（partial
observability）。通常，主体被允许的动作是有限的，例
如，在棋盘中棋⼦只能上、下、左、右移动，或是使⽤的钱
不能多于所拥有的。
强化学习的主体与环境基于离散的时间步作⽤。在每⼀个时 强化学习的典型框架：智能体在环境中采
间 ，主体接收到⼀个观测 ，通常其中包含奖励 。然 取⼀种⾏为，环境将其转换为⼀次回报和
后，它从允许的集合中选择⼀个动作 ，然后送出到环境中 ⼀种状态表⽰，随后反馈给智能体。
去。环境则变化到⼀个新的状态 ，然后决定了和这个变
化 相关联的奖励 。强化学习主体的⽬标，是得到尽可能多的预期回报（return），
其可以为奖励 、带折扣的回报（discounted return） ，其中 为回报因
⼦，表⽰对短期奖励的⿎励。主体选择的动作是其历史的函数，它也可以选择随机的动作。
将这个主体的表现和⾃始⾃终以最优⽅式⾏动的主体相⽐较，它们之间的⾏动差异产⽣了“悔过”
的概念。如果要接近最优的⽅案来⾏动，主体必须根据它的⻓时间⾏动序列进⾏推理：例如，要最
⼤化我的未来收⼊，我最好现在去上学，虽然这样⾏动的即时货币奖励为负值。
因此，强化学习对于包含⻓期反馈的问题⽐短期反馈的表现更好。它在许多问题上得到应⽤，包括
机器⼈控制、电梯调度、电信通讯，AlphaGo（蒙特卡洛树搜索+RL）和星际争霸 II AI
（AlphaStar）[7]。
强化学习的强⼤能⼒来源于两个⽅⾯：使⽤样本来优化⾏为，使⽤函数近似来描述复杂的环境。它
们使得强化学习可以使⽤在以下的复杂环境中：
模型的环境已知，且解析解不存在；
仅仅给出环境的模拟模型（模拟优化⽅法的问题）[8]
从环境中获取信息的唯⼀办法是和它互动。前两个问题可以被考虑为规划问题，⽽最后⼀个问题
可以被认为是genuine learning问题。使⽤强化学习的⽅法，这两种规划问题都可以被转化为机
器学习问题。
策略迭代（Policy Iteration）是RL中策略梯度法的理论基础
值迭代（Value Iteration）与Q-learning存在收敛性等价证明[9]
常⽤算法
蒙特卡洛学习 Monte-Carlo Learning
Temporal-Difference Learning
SARSA算法
Q学习
现代基准补充
算法 环境 性能指标 Year
PPO MuJoCo Humanoid Avg.Reward: 6000 2017
SAC Atari Breakout Max Score: 800+ 2018
R2D2 StarCraft II League Win Rate:72% 2020
探索机制
强化学习需要⽐较聪明的探索机制，直接随机的对动作进⾏采样的⽅法性能⽐较差。虽然⼩规模的
⻢⽒过程已经被认识的⽐较清楚，这些性质很难在状态空间规模⽐较⼤的时候适⽤，这个时候相对
简单的探索机制是更加现实的。
其中的⼀种⽅法就是 -贪婪算法，这种⽅法会以⽐较⼤的概率(1- )去选择现在最好的动作。如果没
有选择最优动作，就在剩下的动作中随机选择⼀个。 在这⾥是⼀个可调节的参数，更⼩的 意味着
算法会更加贪⼼。[10]
前沿⽅向补遗
1. 多智能体RL 引⼊Nash Q-learning框架：
其中 为纳什均衡策略[11]
1. 离线RL(Offline RL) 强调重要性权重约束： πmin E(s,a)∼D [β(a∣s)π(a∣s) Qπ(s,a)] 防⽌分布偏
[12]
移问题
参考⽂献
1. Hu, Junyan; Niu, Hanlin; Carrasco, Joaquin; Lennox, Barry; Arvin, Farshad. Voronoi-Based
Multi-Robot Autonomous Exploration in Unknown Environments via Deep Reinforcement
Learning. IEEE Transactions on Vehicular Technology. 2020-12, 69 (12): 14413-14423.
ISSN 1939-9359. doi:10.1109/TVT.2020.3034800. （原始内容存档于2021-08-13）.
2. Kaelbling, L. P.; Littman, M. L.; Moore, A. W. Reinforcement Learning: A Survey. Journal of
Artificial Intelligence Research. 1996-05-01, 4: 237-285 [2025-03-15]. ISSN 1076-9757.
S2CID 1708582. arXiv:cs/9605103 . doi:10.1613/jair.301. （原始内容存档于2025-05-04）.
3. van Otterlo, Martijn; Wiering, Marco, Wiering, Marco; van Otterlo, Martijn , 编,
Reinforcement Learning and Markov Decision Processes12, Springer Berlin Heidelberg: 3‒
42, 2012 [2025-03-15], ISBN 978-3-642-27644-6, doi:10.1007/978-3-642-27645-3_1
4. 强化学习：原理与Python实现. 北京. 2019: 16‒19. ISBN 9787111631774.
5. Partially Observable Markov Decision Processes, Springer-Verlag, [2025-08-11]
6. Noguer I Alonso, Miquel. Unifying Mathematical Perspectives on Reinforcement Learning:
Integrating Sutton-Barto, Bertsekas and Powell. doi.org. 2025 [2025-08-11].
7. Figure 3: Risk of bias summary (Abreu et al., 2017; Afshar et al., 2010; Ai, 2020; Bolasco et
al., 2011; Cai et al., 2022; Chen, Zhao & Huang, 2019; Dai & Ma, 2021; Deng, 2011; Dong et
al., 2011; Fakhrpour et al., 2020; Fang et al., 2023; Feng et al., 2020; Frih et al., 2017; Hristea
et al., 2016; Jeong et al., 2019; Kozlowska et al., 2023; Leng, 2012; Li et al., 2008; Li & Feng,
2020; Liao et al., 2016; Limwannata et al., 2021; Lu, 2022; Martin-Alemañy et al., 2020, 2016,
2022; Sezer et al., 2014; Shi et al., 2021; Su et al., 2022; Sun, Sun & Yang, 2022a; Tabibi et al.,
2023; Tan et al., 2015; Tayebi, Ramezani & Kashef, 2018; Vijaya et al., 2019; Wang & Liu,
2021; Wang, 2018; Wang et al., 2019; Wang, 2019; Wang et al., 2023; Wei, 2020; Wen et al.,
2022; Wilund et al., 2010; Xu et al., 2022; Xu & Fang, 2016; Yan, Zhao & Peng, 2022; Yang et
al., 2021; Yao et al., 2020; Yu & Cao, 2018; Zeng et al., 2020; Zhou, 2020; Zhou et al., 2016;
Zhu et al., 2020).. doi.org. [2025-08-11].
8. Gosavi, Abhijit. Simulation-based Optimization: Parametric Optimization Techniques and
Reinforcement. Springer. 2003 [2015-08-19]. ISBN 1-4020-7454-9. （原始内容存档于2012-
06-15）.
9. Bertsekas, Dimitri P. Regular Policies in Abstract Dynamic Programming. SIAM Journal on
Optimization. 2017-01, 27 (3) [2025-08-11]. ISSN 1052-6234. doi:10.1137/16m1090946.
10. Tokic, Michel; Palm, Günther, Value-Difference Based Exploration: Adaptive Control
Between Epsilon-Greedy and Softmax, KI 2011: Advances in Artificial Intelligence(PDF),
Lecture Notes in Computer Science 7006, Springer: 335‒346, 2011 [2018-09-03], ISBN 978-
3-642-24455-1, （原始内容存档(PDF)于2018-11-23）
11. Walsh, W. E.; Wellman, M. P. Decentralized Supply Chain Formation: A Market Protocol and
Competitive Equilibrium Analysis. Journal of Artificial Intelligence Research. 2003-11-01, 19
[2025-08-11]. ISSN 1076-9757. doi:10.1613/jair.1213.
12. Levine, Seth M. A comment on Morey et al. (2020). Translational Neuroscience. 2020-01-01,
11 (1) [2025-08-11]. ISSN 2081-6936. doi:10.1515/tnsci-2020-0121.
取⾃“https://zh.wikipedia.org/w/index.php?title=强化学习&oldid=94272891”