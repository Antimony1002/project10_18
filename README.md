# ALOHA 双臂毛巾折叠：NEDF3 → LeRobot ACT

这个目录把原始 I2RT/NEDF3 episode 转为当前 LeRobot Dataset v3，并提供数据检查、ACT
过拟合实验和正式训练入口。原始数据只读；转换结果和训练输出都与原始数据分开。

在 DM3 交互式 H20 实例上训练当前 I2RT 双臂数据，请直接按照
[DM3 ACT 训练说明](docs/dm3_act_training.md)操作。该说明以平台固定的 LeRobot 0.4.4
为准，包含实例选择、持久目录、私有 Hugging Face 数据下载、50 步冒烟测试和五条数据过拟合。
本次实际执行的所有命令、预期输出和故障处理另见
[DM3 ACT 独立操作复现报告](docs/dm3_act_operator_runbook.md)。

## 原始 MCAP / 双腕数据入口

如果数据只有 `metadata.json`、`episode_stats.json` 和 `data/*.mcap`，使用新增的
[I2RT MCAP 转换使用说明](docs/i2rt_mcap_conversion.md)。该入口已按实际采集数据支持
双臂 14 维状态/动作、左右腕视频、A/B/C/D 采集者，以及仅左臂模式；无需导出 NPZ、
真实头部相机或 SuperInference 仓库。预检、转换、验证和 ACT 训练命令均见该说明。

下文原有流程适用于具备 `exports/` 和三路真实相机的数据。

## 已确定的数据契约

每个有效原始 episode 必须包含：

```text
episode_xxxx/
├── metadata.json
├── episode_stats.json
├── data/*.mcap
└── exports/
    ├── joints/{leader,follower}_{left,right}.npz
    ├── grippers/{leader,follower}_{left,right}.npz
    └── videos/{head,left,right}.{mp4 / timestamps_ns.npy}
```

训练帧定义为：

```text
observation.state = follower_left[j0..j5, gripper]
                  + follower_right[j0..j5, gripper]       # 14-D

action            = leader_left[j0..j5, gripper]
                  + leader_right[j0..j5, gripper]         # 14-D

images            = head + left_wrist + right_wrist       # RGB, 480×640
```

head 相机时间戳作为参考时间轴。关节和夹爪按纳秒时间戳线性插值，另外两路相机取最近帧。
转换器会拒绝缺臂、黑色占位相机、低采样率、旧缓存时间戳、NaN、尺寸错误和时间偏差过大的 episode。

`action = leader` 是当前最合理的双边遥操作假设。第一次真机部署前，必须用一条正常数据确认
leader 数值确实是 follower 控制器接收的目标值；若存在额外标定，只改 action 映射层。

## 目录

```text
project/
├── configs/                   # 数据契约与同步阈值
├── data/raw/                  # 原始 episode（不提交、不修改）
├── data/lerobot/              # 转换后的 LeRobot v3
├── deployment/                # 14-D 动作拆分及部署边界
├── outputs/                   # ACT checkpoints
├── reports/                   # 检查报告
├── scripts/                   # 检查、转换、验证、训练入口
├── src/towel_act/             # 转换实现
└── tests/                     # 不依赖机器人硬件的测试
```

## Ubuntu 上的一次性准备

在 LeRobot 仓库根目录按官方方式创建环境；本项目本身不需要额外第三方包。确认下列命令可用：

```bash
uv run python -c "import cv2, numpy, torch, lerobot"
uv run lerobot-train --help
ffmpeg -version
```

后续命令均从 `lerobot/project` 执行。

## 数据一到后的固定流程

### 1. 放置原始数据

可以复制到 `data/raw/`，也可以保留在高速盘并通过 `RAW_ROOT` 指向它：

```bash
RAW_ROOT=/data/towel_folding_raw make inspect
```

不要移动 `_inprogress`、`quarantine` 或仍在写入的 episode。

### 2. 原始数据预检（必须全绿）

```bash
RAW_ROOT=/data/towel_folding_raw make inspect
```

报告写入 `reports/raw_preflight.json`。返回码非零表示至少一条数据不能训练。默认转换器采取严格模式：
只要存在坏数据就停止，避免静默污染训练集。

若坏数据已经明确隔离，可以编辑 `configs/include_episodes.txt`，然后手动执行：

```bash
PYTHONPATH=../src:src uv run python scripts/convert_nedf3_to_lerobot.py \
  /data/towel_folding_raw data/lerobot/towel_folding \
  --config configs/towel_folding.json \
  --episode-list configs/include_episodes.txt
```

### 3. 转换为 LeRobot v3

```bash
RAW_ROOT=/data/towel_folding_raw make convert
```

转换先写入 `data/lerobot/towel_folding.building`，所有 episode 成功并调用 `finalize()` 后才改名为
`data/lerobot/towel_folding`。如果正式输出目录已存在，脚本会停止而不是覆盖。

数据集中的 episode 顺序使用固定随机种子 42 打乱，来源映射保存在：

```text
data/lerobot/towel_folding/conversion_report.json
```

### 4. 验证转换结果

```bash
make validate
```

该步骤会真正通过 LeRobot 读取数据和视频，检查 14-D state/action、三路相机、NaN，并输出：

- episode 数和总帧数；
- batch size 8 时每 epoch 的步数；
- 推荐的 5/10 epoch 训练步数。

随后至少可视化 episode 0，肉眼检查画面与曲线：

```bash
cd ..
uv run lerobot-dataset-viz \
  --repo-id local/towel_folding \
  --root project/data/lerobot/towel_folding \
  --episode-index 0 \
  --save 1 \
  --output-dir project/reports
cd project
```

### 5. 先做 5 条数据过拟合

至少有 5 条有效 episode 后：

```bash
make overfit
```

默认使用 episode 0–4、ACT、60 帧 action chunk、单张 GPU、BF16、10k steps。目标不是泛化，而是确认：

- loss 能持续显著下降；
- 模型可以记住训练轨迹；
- 离线预测的左右臂、夹爪范围合理；
- 部署回放不会出现左右交换、夹爪反向或单位错误。

如果 5 条数据都无法过拟合，停止采更多数据，先排查格式、action 语义和同步。

### 6. 正式 ACT 训练

```bash
STEPS=50000 make train
```

默认使用 batch size 16、10% 完整 episode 留作验证、每 2000 steps 计算验证损失、每 5000 steps
保存 checkpoint。先使用一张 H20；ACT 很小，首次实验用多卡增加的复杂度通常大于收益。

根据 `make validate` 输出，把 `STEPS` 调为约 5–10 epochs，而不是机械使用 50k。
如果已经登录 Weights & Biases，可用 `WANDB_ENABLE=true STEPS=... make train` 开启在线记录；默认关闭，
避免未配置账号时阻塞训练。

## 接下来的研究流程与停止条件

### 阶段 A：采集链路（现在）

1. 修复 `episode_0100` 暴露的设备流问题。
2. 录一条 5 秒测试 episode。
3. `make inspect` 必须显示两臂、两夹爪、三相机采样率全部达标。
4. 回放三路视频并画 leader/follower 曲线。

通过条件：测试 episode 无 error，且相机/关节时间范围重叠。

### 阶段 B：小数据闭环

1. 采 5–10 条固定初始状态、成功的一折演示。
2. 转换、验证、过拟合。
3. 做离线 action 曲线检查。
4. 在低速和急停可用的情况下做第一次真机 rollout。

通过条件：训练集轨迹能被明显复现。否则不扩大数据量。

### 阶段 C：第一版数据集

1. 采 40–60 条成功演示，保持毛巾、桌面、相机和起始位一致。
2. 以完整 episode 做 90/10 train/eval split。
3. 训练约 5 epochs，选择验证损失和真机表现都合理的 checkpoint。
4. 固定条件下测试至少 20 次，记录成功、失败阶段和视频。

通过条件：固定条件成功率足以稳定展示，并能定位主要失败模式。

### 阶段 D：针对失败补数据

只补模型经常失败的状态，例如某一侧毛巾偏移或夹取点偏差。纯失败遥操作数据不进入 ACT
训练集；可以单独保存用于分析。成功恢复轨迹可以进入训练集。

### 阶段 E：最终演示与科研记录

1. 冻结数据版本、配置和 checkpoint。
2. 在固定条件和轻微扰动条件下分别测试。
3. 保存成功率、失败分类、训练曲线和典型视频。
4. README 中记录数据量、采样率、state/action 定义、同步方法、模型参数和已知限制。

## 当前已知限制

- `episode_0100` 是无效链路样本：只有一条旧的左臂/左相机消息，并含右侧和 head 黑图占位流。
- 转换器依赖 NEDF3 自动生成的 `exports/`；原始 MCAP 仍作为不可变备份保留。
- 真机 command sender 尚未实现，因为需要实验室 I2RT 推理接口、关节限位和急停约定。
- 当前不自动覆盖已有数据集。需要重新转换时，请显式换一个输出目录或人工归档旧版本。

## 本地测试

不需要 GPU或机器人：

```bash
make test
```
