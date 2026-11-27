# 诊断：动作增益来自角色，还是来自同角色内对准了人？

对**同一个已经训练好的 mixture**，固定全部候选人的参与概率 q 和动作分布 r。对同角色的两名以上在场候选，把该组的 q 总概率均分到组内，其他组、未知角色和单人组保持原样。比较替换前后的边缘动作 NLL。

从概率上，这等价于固定 q，把每名球员的 r 换成其角色组内的平均 r；也等价于对所有组内随机交换后的**预测概率取平均，再计算 NLL**。它不是对每次随机交换的 NLL 取平均。相比单次轮换，该对照不依赖候选的排列；不改变球队情境或角色之间的总概率分配。

只在有至少两个有效同角色候选的动作上评分，同时报告这批动作占全部可达动作的比例。差值定义为「角色平均 NLL − 原混合 NLL」。正值说明在该批预测里，概率给到原本对应的候选动作分布比同角色平均更好；它不能证明球员有稳定、可跨场复用的个人风格，也不是独立训练的角色基线。

## 无需重训

```bash
python candidate_mixture/role_alignment.py --self-test
python candidate_mixture/role_alignment.py --data local/views/main.npz \
  --run local/runs/main/on-mixture/20260715 \
  --out local/main-role-alignment.json
```

同一实验组的多个种子目录可以在 `--run` 后一起传入。脚本核对训练协议、原数据校验值、事件 ID、原验证 NLL 和可交换样本，按比赛配对重采样。原 checkpoint 若用 GPU 训练，可加 `--device cuda:0`；默认用 CPU。合成用例覆盖了未知角色、单人组、padding 和概率等价关系；现有两个小样本 CPU checkpoint 均通过重放检查。

## 文献到下一步

[Lee 等，Player-aware EPV，§2.4.1、§2.5](https://static.hudl.com/craft/performance-insights-research-stage/2026/Player-aware-EPV-Separating-Skill-from-Tactical-Context-in-Possession-Value-Lee-Hong-Cho-Jeong-Prifti-Bauer-Ko.pdf?v=1795099008) 在已知持球者、使用追踪数据的条件下，用实际选择减去同一比赛情境下共享策略的预测概率，估计个人动作倾向，并按比赛划分估计。可借鉴的是「先扣掉情境，再测个人差异」，不是其追踪特征或数值。后续可在训练期按场交叉拟合角色＋可见情境动作头，再估计向零收缩的球员残差；开发集比较有无残差，两组都使用完全相同且冻结的 q。这样比直接比较个人传球频次更能排除他所面对的比赛局面不同。
