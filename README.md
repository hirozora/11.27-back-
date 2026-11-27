# 动态球员状态与下一动作预测

这里是 [11.27](https://github.com/hirozora/11.27) 的独立实验仓库。首轮模型及实验在 [candidate_mixture](candidate_mixture/README.md)，原始文献与方法对照在 [文献调查](candidate_mixture/LITERATURE.md)。主仓库不需要修改。

研究问题：在给定下一动作球队与时刻、但不知道真实执行者和起点时，从可见事件推断候选球员的参与概率及各自的动作倾向，再检验概率加权能否改善十类动作预测。位置仅作辅助监督。
