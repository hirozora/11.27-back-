# 首轮：对比可见事件归属与概率混合

目标是在同一个 Wyscout 开发协议下回答两个不同的问题：**告诉模型哪些可见事件由候选人执行，能否改善执行者预测？** **按参与概率组合个人动作倾向，能否比直接由比赛情境预测动作更好？**

## 设计

| 实验组 | 候选人拥有的历史事件标记 | 十类动作头 |
|---|---|---|
| `off-context` | 同宽全零 | 直接读球队情境，不读候选人 |
| `on-context` | 每个候选人各有 32 个历史槽位标记 | 直接读球队情境，不读候选人 |
| `off-mixture` | 同宽全零 | 候选人共享动作头，按参与概率加权 |
| `on-mixture` | 每个候选人各有 32 个历史槽位标记 | 候选人共享动作头，按参与概率加权 |

- 保留上游最近 32 条事件、8 组赛前角色和 22 维本场个人履历。赛前个人动作倾向、独立的长个人序列和 Transformer 都留到后续，不同时改变输入与架构。
- 四组模型参数数目、初始化、批序、种子、轮次和选轮指标一致，候选头使用相同维数；没有 ID embedding。主视图保留 0–2 秒内符合上游条件的己方有球事件；延迟视图只用严格早于 2 秒的事件。两视图各自同样本比较。
- 默认训练损失是执行者 NLL + 真执行者条件动作 NLL + 九宫格辅助 NLL。真实执行者只用于索引损失，真实起点只进入辅助损失。推断的动作概率是 `sum_i q_i * r_i`。默认按**边缘动作 NLL**选 checkpoint；如需先隔离关系对执行者的贡献，可对两组同步使用 `--event-weight 0 --select actor_nll`。
- 在同一拟合好的 mixture 上诊断软混合、Top-1 硬选择（并列时对并列候选等权）、均匀权重、真执行者和同角色调换动作分布。调换时固定 q，仅交换已知同角色候选的 r，只在可交换的样本统计。它检验候选与动作分布的对应关系，不能单独证明模型学到了个人稳定风格。真执行者诊断不是正式成绩，也不保证优于软混合。
- 主指标是动作边缘 NLL，另报宏 F1、十类混淆矩阵、执行者 NLL/Top-1/Top-3、条件区域 NLL 和候选覆盖率。评价器检查每组使用同一数据和同一验证行。区间用比赛为单位配对重采样，对动作加权；开发集也用于选轮，不是最终测试。
- 所有组都只在可达样本评分，报告分母及覆盖率；当前不主张代表全部事件。所给球队和查询时刻是共同条件，不是无条件的逐事件模拟。

## 准备

以下命令从本仓库根目录运行。建议使用 [上游已核对提交](https://github.com/hirozora/11.27/tree/0323a6d37bdef72dbc57eed50d6a058e648cae61)。需要上游的 Python 依赖、PyArrow 和 PyTorch 2.6。原始记录按[上游说明](https://github.com/hirozora/11.27#使用方法)准备，可从 [Wyscout 数据发布页](https://doi.org/10.6084/m9.figshare.c.4415000.v2)获取。归属位不能从旧 NPZ 还原。

```bash
# 只需在没有现成 processed/England_nontest 时执行。
(
  cd ../11.27/scene_regularity_release
  python src/build_actions.py --config configs/data/England_nontest.json
  python src/build_model_data.py --config configs/model/England_nontest.json
)
UP=../11.27
DATA="$UP/scene_regularity_release/data/processed/England_nontest"
python candidate_mixture/test_protocol.py --upstream "$UP"
for view in main lag; do
  python candidate_mixture/build_views.py --upstream "$UP" --data "$DATA" --out local/views --view "$view"
done
```

如果本地有原上游的未修改 NPZ，可加 `--reference /path/to/main.npz`（另一个视图用 lag）逐数组校对原始特征。生成的新数据只写到 `local/`，不覆盖上游。

## 先跑什么

先在主视图跑一个种子，四组共用一次建表。下面是完整的 120 轮单种子试跑，不需要碰时间测试集；初看脚本流程可另建输出目录加 `--epochs 1`，不要把它的分数当研究结果。

```bash
UP=../11.27
seed=20260715
for arm in off-context on-context off-mixture on-mixture; do
  python candidate_mixture/train.py --upstream "$UP" --data local/views/main.npz \
    --out "local/runs/main/$arm/$seed" --arm "$arm" --seed "$seed" --device cuda:0
done
python candidate_mixture/evaluate.py --runs local/runs/main --seeds "$seed" \
  --out local/main-pilot.json
```

先看 `on-mixture_vs_on-context.event_nll`（两者之差量化同样使用关系位时，候选混合相对直接分类的收益）和 `on-mixture_vs_off-mixture.actor_nll`（检验归属标记是否帮助分辨执行者）。JSON 中成对差值为正表示改善。若只改善后者，优先看软混合相对均匀权重、真执行者与同角色调换的差距，不急着上 Transformer。

然后用不变的配置补上 `20260716`、`20260717`，在延迟视图同样跑三个种子，分析每个视图内部的配对差值。`evaluate.py` 默认使用三个种子。正式时序测试要等模型、指标和规则冻结后再单独跑。
