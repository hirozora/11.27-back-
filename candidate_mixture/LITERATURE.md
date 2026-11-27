# 动态球员状态、下一执行者和动作类型：首轮文献调查

**可验证的研究问题：** 给定目标球队和时刻，但不给真实执行者、目标起点或动作，从可见事件估计同队在场候选人的动态状态，再检验利用候选人概率是否能改善下一动作类别预测。这是条件预测，不是同时预测球队和时刻的完整比赛生成。

## 直接相关

- **Romero 等，MLSA 2025，§4、表 2。** [Next-Event Prediction in Soccer: Assessing the Impact of Team and Player Information](https://dtai.cs.kuleuven.be/events/MLSA25/papers/MLSA25_paper_353.pdf)。Wyscout 历史窗口预测下一动作类别。MLP 将最近事件身份/角色特征与历史动作拼接；3 条历史时，基线 F1 0.272、球队+角色 0.304、全特征 0.276；同三组 XGBoost 为 0.312、0.313、0.307。不能说球员信息没用：角色有时有效，静态 ID 增益不稳定。可做同协议的静态特征对照。
- **Someya 等，JSAI 2023，§3.2、表 2。** [Leveraging Player Embedding for Soccer Event Prediction](https://www.jstage.jst.go.jp/article/pjsai/JSAI2023/0/JSAI2023_2M5GS1005/_pdf/-char/en)。动作标签历史只到 t−1，但输入包括 t 的真实球员和真实起点；加入球员向量后测试宏 F1：GRU 49.20→51.73，Transformer 44.80→48.01。是已知球员和位置时的支持证据，不能作为严格窗口的对比成绩。
- **Hong 等，EventGPT，arXiv:2512.17266v3，§2.2。** [Capturing Player Impact from Team Action Sequences Using GPT-Based Framework](https://arxiv.org/html/2512.17266v3)。英超 SPADL 序列预测类型等属性及 rOBV；目标球员身份是输入条件，明确不预测身份；反事实场景中替换身份。不是未知真实执行者时的候选概率加权。
- **Hong 等，ScoutGPT，ECML-PKDD 2026 Applied Data Science Track，arXiv:2603.15212v2，§3.3。** [Modeling Matches as Language: A Generative Transformer Approach for Counterfactual Player Valuation in Football](https://arxiv.org/html/2603.15212v2)。K 联赛 VERSA，由生成的球队、角色缩小候选，同角色多位球员时，以最近球位置（上一动作终点）到训练期平均位置的距离硬选，连续控球时锁定球员。是强 actor-first 先例，不能声称首个先找人再预测动作；本项目的区别是候选人真实标签监督和全名单动作概率混合。可加最近可见本队执行者等低成本概率基线。
- **代表性事件序列基线。** [LEM](https://link.springer.com/article/10.1007/s10994-024-06606-y) 已用下一动作类型条件化后续属性；[Seq2Event 原作者代码与论文入口](https://github.com/statsonthecloud/Soccer-SEQ2Event) 用 RNN/Transformer 编码事件流。属性分解不算创新，情境直接分类是必要对照。

## 可转化的方法

- [Player Vectors（2019）](https://ecmlpkdd-storage.s3.eu-central-1.amazonaws.com/former-websites/2019/downloads/paper/701.pdf) 和 [SoccerMix（2020）](https://ecmlpkdd-storage.s3.eu-central-1.amazonaws.com/preprints/2020/lncs12461/lncs12461465.pdf) 用事件描述某人在哪些位置采取哪些动作，后者还区分方向。可转化为按球员、区域、动作的赛前计数，对小样本向同角色均值平滑，按训练时间计算并冻结。SoccerMix 不是按执行者混合动作预测。
- [Pointer Networks（2015）](https://proceedings.neurips.cc/paper_files/paper/2015/file/29921001f2f04bd3baee84a12e98098f-Paper.pdf) 支持按动态候选集做 masked Softmax；[Adaptive Mixtures of Local Experts（1991）](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf) 提供输入相关概率和的建模先例。这里 q 有真实执行者监督，动作头由所有候选共享；公式本身不算创新。
- [Fernández 等，EPV（2021）](https://link.springer.com/article/10.1007/s10994-021-05989-6) 在已知持球者、具备追踪信息的场景按动作概率汇总动作价值，也按潜在落点汇总传球价值；这是足球内部的概率分解先例，不是候选执行者预测。
- [JODIE（2019）](https://snap.stanford.edu/jodie/) 根据时序互动更新实体状态，并考虑上次互动后的时间差。可用于后续候选专属记忆，但第一轮不必上图或 RNN。[足球 TGN 先例](https://link.springer.com/article/10.1007/s10994-024-06611-1) 使用 25 Hz 追踪数据与事件图预测链接及动作；[Agent Imputer（AAMAS 2023）](https://www.ifaamas.org/Proceedings/aamas2023/pdfs/p1643.pdf) 在稀疏事件下补全全员位置，但训练需追踪真值，且用当前事件和整场统计，不适用于严格过去窗口。

## 首轮实验对应

1. **关系。** 在现有可见窗口的相同索引取执行者 ID，做候选人×事件槽相等判断。屏蔽无效候选、历史 padding 和未知历史执行者；不增加全局 ID embedding 或使用目标字段。保留槽位次序，关闭关系时送相同维度的全零输入。对现有主视图、两秒延迟视图分别建表。
2. **训练和推断。** 主损失为执行者 NLL + 真执行者的条件动作 NLL + 固定权重位置辅助 NLL。目标执行者只索引损失，不进编码；正式预测在候选维计算 logsumexp(log q + log r)。单独加边缘动作 NLL 可作消融，不宜只用该损失，否则当 r 全部相同时 q 不可识别。九宫格是辅助，不是最终研究目标。
3. **消融。** 同 MLP 和预算做关系 off/on；用同一 q、r 对比软汇总、硬 top-1 和真实执行者诊断（不是正式预测，也不是数学保证的上界）。增加同预算情境直接分类头。要论证个人而非角色，在有两名及以上同角色候选的样本固定 q、仅交换 r 的候选个人输入块；反向可只交换组内 q，同时保持角色总概率不变。均匀 q 会同时破坏角色和个人信息，不能单独支撑个人性论断。
4. **评估。** 主报边缘动作 NLL，辅报动作宏 F1、重点类混淆、执行者 NLL/Top-k、条件动作 NLL、候选覆盖率。相同样本和种子按场配对 bootstrap，开发集选模型，时间测试只在规则冻结后评一次。不可达样本不能悄悄排除：同集比较并报告覆盖率，若可做全样本动作预测，另表报告。
5. **升级规则。** 若关系改善 q 但不改善动作边缘，优先诊断不同读出和 r，而非直接换 Transformer；有边缘增益之后再匹配信息和容量比较 Transformer。赛前倾向、个人序列和图结构应独立加入。

英文定位：*We study whether candidate-specific, time-varying player states built from visible event histories improve next-action classification when the next actor is unknown. Given the target team and query time, we supervise a distribution over eligible actors and marginalize an actor-conditioned action distribution over that roster.* 不据本轮检索宣称首个球员嵌入、首个 actor-first 或首创概率混合。

### 什么时候才可以加入传球终点、接球人和 SPADL 事件？

[ELASTIC §2.1](https://arxiv.org/html/2608.30227v2)解决的是离线事件与追踪对齐。它在相邻事件之间插入接球事件时，可以使用后一事件的执行者。[socceraction 的 Wyscout 转换器](https://github.com/ML-KULeuven/socceraction/blob/master/socceraction/spadl/wyscout.py)也会用后续越位记录修正较早的传球，并插入合成带球。这些是合理的离线标注，不天然是预测时可用的输入。给原事件加两秒延迟不保证其派生字段已知。如果以后纳入它们，应在原始层截断或扰动所有未到可见阈值的记录，重新转换，检查每个查询的输入不变；只有全部依赖记录都可见时才使用相应派生字段。这里没有发现现有起点及归属位做法有此问题。
