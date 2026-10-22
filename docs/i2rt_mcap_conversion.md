# I2RT 原始 MCAP 转换为 LeRobot ACT

此入口直接读取正式 NEDF3 Episode 的 MCAP 分片，不要求 `exports/`、
`dm3-core`、`nmx-msg`、SuperInference 仓库、WebUI 或机器人硬件。
兼容 `i2rt-nedf3-v2` 的 RGB JPEG 契约，使用官方 LeRobot 0.4.4 写 Dataset v3。
原来的 NPZ/三相机转换入口仍可使用；这两种输入分别使用各自的转换与验证命令。

## 1. 创建独立环境

以下命令从本仓库根目录执行，建议 Python 3.12；不要安装到设备采集环境中。

```bash
uv venv .venv-i2rt --python 3.12
uv pip install --python .venv-i2rt/bin/python --torch-backend cpu \
  -r requirements-i2rt-mcap.txt
```

CPU 环境支持转换、验证和 ACT 兼容性检查。GPU 训练机需自行选择匹配硬件的
Torch 后端，保持 LeRobot 0.4.4。运行以下测试时额外安装 pytest：

```bash
uv pip install --python .venv-i2rt/bin/python pytest
PYTHONPATH=src .venv-i2rt/bin/python -m pytest tests -q
```

## 2. 放置原始数据并选择采集者

推荐每人一个目录，同一人的不同任务可继续嵌套：

```text
data/raw/A/任务名/episode_0000/
  metadata.json
  episode_stats.json
  data/mcap_index.json
  data/*.mcap
data/raw/B/任务名/episode_0000/
```

使用 `configs/i2rt_mcap_sources.yaml` 指定来源。路径相对于清单本身；
数据也可保留在外置盘，填入实际的绝对源目录。启用 B/C/D 条目前应确保对应目录存在。
采集者 A/B/C/D 标签写入溯源信息，不作为模型输入；同名 Episode 在不同目录下分别保留。
默认保留新目录的 0000、0001，不继承其他历史目录的丢弃规则。

```yaml
version: 1
sources:
  - path: ../data/raw/A
    collector: A
  - path: ../data/raw/B
    collector: B
```

可追加 `--collectors A` 只转换清单中的 A。它只筛选标签，不能区分同属 A 的
新旧根目录；是否包含旧数据由清单中的 `path` 决定。
正式 Episode 的原始文件保持只读，`_inprogress` 和 `quarantine` 不参与转换。

## 3. 确认动作映射与训练输入

将采集当时使用的完整遥操作 YAML 传给 `--teleop-config`，或按其真实映射填写
`configs/i2rt_teleop_directions.yaml`。示例为左右全 +1，不能盲目套用到镜像关节。
第七维是归一化夹爪，只允许 +1。每批映射发生变化时分别转换，并保存相应配置。

| 特征 | 双臂默认内容 |
| --- | --- |
| `observation.state` | follower 左六个关节角 + 左夹爪，再右六个关节角 + 右夹爪，14 维 |
| `action` | 同样顺序的 leader 位置，应用采集时 follower 关节符号，14 维 |
| `observation.images.left_wrist` | 左腕 RGB，原生 640×480，H.264 |
| `observation.images.right_wrist` | 右腕 RGB，原生 640×480，H.264 |
| 时间与单位 | 统一 30 Hz，关节 rad、夹爪开度 0–1；有界插值低维流，取最近相机帧 |

仅左臂时使用 `--arms left --cameras left`，得到 7 维状态/动作和左腕画面。
此入口不要求真实头部相机；主动选择合成黑色头部会被拒绝。
动作表示遥操作激活时的从臂目标。源数据没有激活/安全保持标签，需人工确认有效示教范围。

## 4. 预检、转换、验证

先预检，检查列出的 Episode 是否属于本批；预检不创建数据集：

```bash
.venv-i2rt/bin/python scripts/convert_i2rt_mcap_to_lerobot.py \
  --manifest configs/i2rt_mcap_sources.yaml \
  --teleop-config configs/i2rt_teleop_directions.yaml \
  --arms left right --cameras left right --fps 30 --gap-policy reject --dry-run
```

再指定一个尚不存在的输出目录：

```bash
.venv-i2rt/bin/python scripts/convert_i2rt_mcap_to_lerobot.py \
  --manifest configs/i2rt_mcap_sources.yaml \
  --teleop-config configs/i2rt_teleop_directions.yaml \
  --arms left right --cameras left right --fps 30 --gap-policy reject \
  --output data/lerobot/i2rt_a_dual_v1 --repo-id local/i2rt_a_dual_v1

.venv-i2rt/bin/python scripts/validate_i2rt_lerobot.py \
  data/lerobot/i2rt_a_dual_v1 --repo-id local/i2rt_a_dual_v1 --act-smoke
```

验证使用官方 LeRobot 加载器，检查每条 Episode 首、中、尾的两路画面、全部低维行及
归一化夹爪范围。`--act-smoke` 额外运行 60 步动作块的小型随机 ACT 前向和反向，
检查有限损失与梯度，不执行正式训练。随机骨干无需下载预训练模型。

数据集的 `data/` 是 Parquet，`videos/` 是两路视频，`meta/stats.json` 是归一化统计。
v3 会把多个 Episode 合并进同一数据文件，Episode 数以加载器和 `meta/episodes/` 为准。
`meta/i2rt_sources.json` 保存采集者、源路径、关节符号、配置哈希、原始元数据哈希与分段范围；
`meta/source_timestamps/` 保存每帧的原始纳秒时间戳。`--repo-id` 只是本地标识，不触发上传。

## 5. 筛选、裁剪与失败处理

- 在源条目添加 `exclude: [episode_0003]` 排除指定名称；它作用于该条目下所有任务，
  所以仅针对某个任务时将 `path` 缩小至任务目录。
- 单条裁剪使用精确 Episode 路径，添加 `start_s: 2`、`end_s: 20`，单位为录制开始后的秒。
  用该条目替换原目录条目，避免同一 Episode 重复分配；目录级裁剪会应用于其全部 Episode。
- 默认拒绝 MCAP CRC/计数不符、写入错误、队列丢弃、NaN、缺流、非归一化夹爪和错误尺寸。
  最大源间隔 200 ms；最近样本偏差上限为关节 25 ms、夹爪 60 ms、相机 75 ms。
- 时间断档时先检查示教。人工审查后可显式 `--gap-policy split`，只保留至少 3 秒的连续段，
  不能把动作块插值跨过断档。同一原始 Episode 的多个段应放在相同训练/验证分组。
- 构建先写入 `<output>.building`，所有数据完成并关闭 writer 后才改为正式目录。
  失败目录保留供排查，不可直接改名充当训练集；更换新输出路径重试。
- 每次完整转换读取当前清单中的全部数据，不追加、不覆盖已有数据集。新增数据后先核对
  预检列表，再使用新版本输出路径；验证和训练的 root、repo-id 随之更新。
- 如果导入或版本检查失败，在独立环境中核对上述依赖，不回退到设备 Python。

## 6. ACT 训练入口

以下是双臂、双腕的离线 CPU 训练示例，执行会开始 1000 步正式训练；
GPU 机器请使用自己的训练环境和 `--policy.device=cuda`。小数据接口通过不代表策略质量。

```bash
.venv-i2rt/bin/python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/i2rt_a_dual_v1 --dataset.root=data/lerobot/i2rt_a_dual_v1 \
  --dataset.video_backend=pyav --policy.type=act --policy.device=cpu \
  --policy.pretrained_backbone_weights=null --policy.push_to_hub=false \
  --policy.chunk_size=60 --policy.n_action_steps=60 \
  --batch_size=8 --num_workers=0 --steps=1000 --eval_freq=0 \
  --output_dir=outputs/act_i2rt_a_dual_v1 --wandb.enable=false
```

该命令从随机骨干开始，不上传数据或模型。输出 root 和 repo-id 需与转换时一致。
不要对双腕数据使用旧的“必须三相机”的验证脚本；请使用本页的 `validate_i2rt_lerobot.py`。
