# 等预算的执行者—区域归属位实验

固定主仓库 `cb165beb` 的执行者—区域模型、数据及预算。唯一开关是向候选球员隐藏层加入 32 位的历史事件归属向量，或把同样的输入置零。投影无偏置且零初始化，公共网络与区域头沿用原初始化。两组从头训练，共用种子、每场轮转抽样与开发执行者 NLL 选轮。使用主仓库的数据字段逐项核对重建视图，新增归属位不使用目标。默认 120 轮，每场每轮 128 条，batch 256，学习率及梯度裁剪沿用主仓库。时间留出测试不参与。

本轮主诊断是严格延迟视图，种子 20260715。两组都保存逐候选执行者概率；评估时只代入这些概率，区域条件分布和个人／位置组动作头保持归档参数，不使用各新模型区域头的输出。这样先回答「仅执行者估计变化是否改善动作」，不同于比较整体联合模型。比较归属开关时用等预算的两组，另用正式归档作为绝对性能参照。分解见 [组间／组内贡献](CANONICAL_BRIDGE.md)。

## 复算

先按 [桥接实验](CANONICAL_BRIDGE.md#复算)准备主仓库归档视图与含归属窗口的备用视图。示例从备用仓库根目录运行；有 GPU 可把 `--device cpu` 换成 `--device cuda:0`（CPU 与 GPU 不保证浮点逐位相等）。

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
c=../11.27/onball_participation
for mode in off on; do
  python candidate_mixture/train_joint_ownership.py --canonical "$c" \
    --data real-views/lag.npz --reference "$c/results/lag.npz" \
    --out "joint-ownership/$mode" --ownership "$mode" --seed 20260715 --device cpu
done
python candidate_mixture/canonical_bridge.py --joint-ownership --canonical "$c" \
  --data "$c/results/lag.npz" --matches "$c/results/event13/inputs/matches.parquet" \
  --archive "$c/results/event13/lag_20260715" --off joint-ownership/off \
  --on joint-ownership/on --out joint-ownership/bridge.json
```
