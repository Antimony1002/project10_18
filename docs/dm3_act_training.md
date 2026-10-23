# DM3 上训练 I2RT 双臂 ACT

本页针对当前的 LeRobot v3 数据集和 LeRobot 0.4.4。第一次只申请一张 H20，先完成
小数据过拟合；不要一开始使用四卡或分布式训练。

## 1. 创建交互式实例

在“模型训练 → 交互式训练”中创建实例：

- 训练团队：本组团队；
- GPU：H20 × 1；
- 优先级：低；卡满时排队：开启；
- 镜像：Python 3.12、Ubuntu 22.04、CUDA 12.6、PyTorch 2.7.1 的训练镜像；
- 有效期：首次可设 1 天；
- 资产挂载：首次可留空。当前私有 Hugging Face 数据集只有约 100 MB，登录实例后下载到
  `~/output/data` 更直接。

创建后等待状态变为“运行中”，记录页面给出的 SSH 主机、端口和 Linux 用户名。先执行
`nvidia-smi`，确认只有申请的一张 H20 可见。

## 2. 持久目录与代码

```bash
mkdir -p ~/output/{code,data,envs,runs}
cd ~/output/code
git clone https://github.com/Antimony1002/project10_18.git
cd project10_18
```

`~/output` 是需要保留的持久目录；不要把代码、数据或 checkpoint 只放在容器的临时路径。

## 3. Python 环境

复用镜像里的 CUDA PyTorch，避免重复下载或替换驱动匹配的 Torch：

```bash
python3 -m venv --system-site-packages ~/output/envs/act
source ~/output/envs/act/bin/activate
python -m pip install --upgrade pip
python -m pip install "lerobot==0.4.4"
python -c "import torch, lerobot; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name())"
```

如果安装日志准备替换镜像自带的 Torch，先停止并核对版本；LeRobot 0.4.4 接受
Torch 2.2.1–2.10，截图所示 Torch 2.7.1 在范围内。

## 4. 下载私有 Hugging Face 数据

在实例中使用一个仅有该 Dataset 读取权限的 fine-grained token 登录；不要把 token 写入仓库或命令历史。

```bash
HF_ENDPOINT=https://huggingface.co hf auth login
HF_ENDPOINT=https://huggingface.co hf download \
  Hydrogen1002/i2rt_a_dual_act_20261022_all_0000_0015 \
  --repo-type dataset \
  --local-dir ~/output/data/i2rt_a_dual_act_20261022_all_0000_0015
```

确认 `meta/info.json`、Parquet 和两路 MP4 都存在：

```bash
find ~/output/data/i2rt_a_dual_act_20261022_all_0000_0015 -maxdepth 3 -type f | sort
```

## 5. 数据与 ACT 冒烟检查

```bash
cd ~/output/code/project10_18
PYTHONPATH=src ~/output/envs/act/bin/python scripts/validate_i2rt_lerobot.py \
  ~/output/data/i2rt_a_dual_act_20261022_all_0000_0015 \
  --repo-id Hydrogen1002/i2rt_a_dual_act_20261022_all_0000_0015 \
  --act-smoke
```

该命令必须报告 16 episodes、14,141 frames、14 维状态/动作、左右腕两路相机，并完成有限
loss 和 backward。失败时不要开始正式训练。

## 6. 先跑 50 步，再做五条数据过拟合

先验证训练链路和显存：

```bash
STEPS=50 BATCH_SIZE=2 NUM_WORKERS=0 bash scripts/train_overfit.sh
```

检查 `~/output/runs/` 下产生配置、日志与 checkpoint，且 loss 为有限数。然后运行真正的小数据过拟合：

```bash
STEPS=10000 BATCH_SIZE=8 NUM_WORKERS=8 bash scripts/train_overfit.sh 2>&1 | \
  tee ~/output/runs/act_overfit_console.log
```

默认只使用 episodes 0–4，随机初始化 ResNet18，不依赖外部预训练权重。目标是 loss 持续明显下降，
证明数据键、动作语义、视频解码和模型反向传播形成闭环；这不是最终泛化模型。

## 7. 正式训练暂缓

当前 16 条数据的任务标签为“测试”，适合验证流水线，不足以代表毛巾折叠任务。完成五条过拟合后，
应先采集并冻结正式毛巾折叠数据，再运行：

```bash
STEPS=8000 BATCH_SIZE=16 bash scripts/train_act.sh
```

14,141 帧在 batch size 16 下约为 884 steps/epoch，8,000 steps 约为 9 epochs。LeRobot 0.4.4
没有本项目旧脚本所假设的离线 `dataset.eval_split`；正式实验应按完整 episode 单独划分验证集，
并用独立评估脚本计算验证损失，不能把帧随机拆分。
