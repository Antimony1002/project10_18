# DM3 上独立完成 ACT 训练：操作复现报告

> 适用环境：DM3 交互式训练、H20、Ubuntu 22.04、Python 3.12、CUDA 12.6、
> PyTorch 2.7.1、LeRobot 0.4.4。本文记录 2026-10-23 实际跑通的命令。

## 1. 这份手册解决什么问题

按照本文操作，即使没有他人远程协助，也可以独立完成：

1. 在 Mac 上配置 SSH；
2. 创建 DM3 GPU 实例；
3. 建立持久工作目录和隔离环境；
4. 把 LeRobot 数据送入实例；
5. 验证数据和 ACT 前后向传播；
6. 运行 50-step 冒烟训练；
7. 在后台运行 10k-step 小数据过拟合；
8. 查看 loss、显存、进程和 checkpoint；
9. 停止实例并保留结果。

本手册只验证训练闭环。它不包含 ALOHA 真机控制器、动作安全限幅和在线 rollout。

## 2. 已验证的数据与结果

本次数据集：

```text
Hydrogen1002/i2rt_a_dual_act_20261022_all_0000_0015
```

数据契约：

- LeRobot Dataset v3.0；
- 16 episodes，14,141 frames，30 Hz；
- `observation.state`：14 维；
- `action`：14 维；
- 相机：`left_wrist`、`right_wrist`；
- 本次过拟合只取 episodes 0–4。

实测结果：

- 50-step 冒烟训练成功；
- 完整 ACT 约 52M 参数；
- 10,000 steps、batch size 8 用时约 10 分 34 秒；
- loss 从 step 50 的 15.398 降到 step 10,000 的 0.127；
- 2k、4k、6k、8k、10k checkpoint 均成功保存；
- 最终 `model.safetensors` 约 206 MB；
- 含优化器状态的完整 10k checkpoint 约 597 MB。

loss 明显下降说明模型可以记忆这五条轨迹，即数据读取、动作维度、视频解码、ACT、CUDA、
反向传播和保存路径已经闭环。它不等价于真机泛化成功。

## 3. Mac 上配置 SSH

先检查是否已有公钥：

```bash
ls -l ~/.ssh/*.pub
```

若已有 `~/.ssh/id_ed25519.pub`，复制它：

```bash
pbcopy < ~/.ssh/id_ed25519.pub
```

若没有，再生成：

```bash
ssh-keygen -t ed25519 -C "dm3"
pbcopy < ~/.ssh/id_ed25519.pub
```

在 DM3 右上角账户设置中，把这一整行粘贴到 SSH Key 并保存。只上传 `.pub` 公钥，绝不能
上传 `~/.ssh/id_ed25519` 私钥。

## 4. 创建 DM3 交互式实例

入口：`模型训练 → 交互式训练 → 单机实例 → 创建实例`。

推荐配置：

| 配置项 | 选择 |
| --- | --- |
| 训练团队 | 本科生实习考核 |
| 调试模式 | 关闭 |
| GPU | H20 × 1 |
| CPU / 内存 | 默认 |
| 优先级 | 低 |
| 卡满时排队 | 开启 |
| 镜像来源 | Harbor 镜像 |
| 一级 | `pretrain` |
| 仓库 | `nmx_vla` |
| Tag | `nmx-vla-torch271-py312-cuda126-ubuntu2204-20260313_` |
| 资产挂载 | 小数据首次验证可留空 |
| 有效期 | 1 天 |

完整镜像名：

```text
harbor.noematrix.cn/pretrain/nmx_vla:nmx-vla-torch271-py312-cuda126-ubuntu2204-20260313_
```

ACT 单卡即可。即使实例暴露多张卡，以下命令也会用 `CUDA_VISIBLE_DEVICES=0` 固定到一张卡。
不要因为能看到多张 H20 就默认多卡训练；当前脚本没有启动 DDP。

## 5. SSH 登录与硬件检查

实例状态变为“运行中”、Linux 账号显示“已就绪”后，点击实例的 SSH 按钮，复制平台给出的
命令。形式如下：

```bash
ssh <用户名>@<SSH主机> -p <SSH端口>
```

第一次连接会询问主机指纹，核对是 DM3 页面提供的主机和端口后输入：

```text
yes
```

登录后检查：

```bash
nvidia-smi
python3 --version
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

本次验证环境应看到 Python 3.12、Torch 2.7.1+cu126，且 CUDA 可用。

## 6. 建立持久目录并克隆代码

以下命令都在远程 DM3 终端执行：

```bash
mkdir -p ~/output/{code,data,envs,runs}
cd ~/output/code
git clone https://github.com/Antimony1002/project10_18.git
cd project10_18
```

如果仓库已经存在，不要重复 clone：

```bash
git -C ~/output/code/project10_18 pull --ff-only
```

目录用途：

```text
~/output/code   Git 代码
~/output/data   自行复制或下载的数据
~/output/envs   Python 虚拟环境
~/output/runs   日志、配置和 checkpoint
```

代码、数据和模型必须放在 `~/output` 下。实例停止后这些内容仍保留；不要把重要结果只放在
`/tmp` 或容器其他临时位置。

## 7. 创建经过验证的 Python 环境

镜像本身带 CUDA 版 PyTorch，因此使用 `--system-site-packages` 复用它，避免重新下载几 GB
的 Torch/CUDA wheel：

```bash
python3 -m venv --system-site-packages ~/output/envs/act
source ~/output/envs/act/bin/activate
python -m pip install --upgrade pip
```

### 7.1 当前镜像的快速安装方式

当前镜像已经提供大部分依赖，但系统层是 LeRobot 0.3.3、PyAV 16。使用下面的精确版本在
虚拟环境中覆盖它们：

```bash
python -m pip install \
  --index-url https://mirrors.aliyun.com/pypi/simple/ \
  --no-deps \
  "numpy==1.26.4" \
  "lerobot==0.4.4" \
  "datasets==4.8.5" \
  "diffusers==0.35.2" \
  "huggingface-hub==0.35.3" \
  "av==15.1.0"
```

这里使用 `--no-deps` 是针对上述指定镜像的实测优化，避免下载训练用不到的 CMake、机器人硬件
和可视化组件。若将来换了基础镜像，不应照搬 `--no-deps`；改用完整安装并重新验证：

```bash
python -m pip install "numpy<2" "lerobot==0.4.4"
```

检查最终生效的是虚拟环境版本：

```bash
CUDA_VISIBLE_DEVICES=0 ~/output/envs/act/bin/python - <<'PY'
import av
import datasets
import lerobot
import numpy
import torch
from lerobot.scripts import lerobot_train

print("numpy", numpy.__version__)
print("torch", torch.__version__)
print("lerobot", lerobot.__version__)
print("datasets", datasets.__version__)
print("av", av.__version__)
print("cuda", torch.cuda.is_available(), torch.cuda.device_count())
print("gpu", torch.cuda.get_device_name(0))
print("train_import_ok")
PY
```

预期至少包含：

```text
numpy 1.26.4
torch 2.7.1+cu126
lerobot 0.4.4
av 15.1.0
cuda True 1
train_import_ok
```

## 8. 把数据送入实例

### 8.1 小数据：从 Mac 直接 SCP

本次实测 DM3 无法连接 `https://huggingface.co/api/whoami-v2`，会报
`Network is unreachable`。这不是 token 错误。在网络恢复前，小数据直接从 Mac 复制最稳。

在 **Mac 新终端** 执行，主机、端口和用户名替换为 DM3 页面当前显示的值：

```bash
scp -P <SSH端口> -r \
  /Users/donglujia/lerobot/project/data/lerobot/i2rt_a_dual_act_20261022_all_0000_0015 \
  <用户名>@<SSH主机>:/home/users/<用户名>/output/data/
```

远程镜像没有 `rsync`，所以本次使用 `scp`。复制完成后，在远程终端检查：

```bash
DATASET_ROOT=~/output/data/i2rt_a_dual_act_20261022_all_0000_0015
find "$DATASET_ROOT" -type f | wc -l
du -sh "$DATASET_ROOT"
test -f "$DATASET_ROOT/meta/info.json" && echo "info.json OK"
```

本次应看到 24 个文件。Mac 的 APFS 和远程文件系统对稀疏/块大小的统计不同，`du` 显示的总大小
不必逐字节相同；文件结构和后续 LeRobot 解码检查更重要。

### 8.2 大数据：DM3/OSS 资产挂载

正式毛巾数据增长到几十或几百 GB 后，不应每次从 Mac 上传：

1. 把冻结的数据版本上传到实验室 OSS；
2. 在 DM3“数据集 → 外部资产”登记；
3. 创建实例时在“资产挂载”选择它；
4. 登录后从 `/mnt/dataset/<slug>` 只读访问；
5. checkpoint 和日志仍写入 `~/output/runs`。

挂载数据是只读的。需要转换或预处理时，将新生成结果写到 `~/output/data`，不要尝试修改
`/mnt/dataset`。

### 8.3 Hugging Face 仅在官方接口可达时使用

如果下面的命令能返回 HTTP 状态而不是连接超时，才走 Hugging Face：

```bash
curl -I --connect-timeout 10 https://huggingface.co
```

然后执行：

```bash
HF_ENDPOINT=https://huggingface.co hf auth login
HF_ENDPOINT=https://huggingface.co hf download \
  Hydrogen1002/i2rt_a_dual_act_20261022_all_0000_0015 \
  --repo-type dataset \
  --local-dir ~/output/data/i2rt_a_dual_act_20261022_all_0000_0015
```

登录询问 `Add token as git credential?` 时输入 `n`。数据大小与 Git credential 无关；不要把 token
写入脚本、Git 仓库或聊天记录。`hf-mirror.com` 匿名连接可达不代表它能可靠认证私有仓库，未验证前
不要把镜像当作私有数据的唯一传输路径。

## 9. 数据与 ACT 前后向验证

在远程终端执行：

```bash
cd ~/output/code/project10_18
DATASET_ROOT=~/output/data/i2rt_a_dual_act_20261022_all_0000_0015

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
  ~/output/envs/act/bin/python scripts/validate_i2rt_lerobot.py \
  "$DATASET_ROOT" \
  --repo-id Hydrogen1002/i2rt_a_dual_act_20261022_all_0000_0015 \
  --act-smoke
```

通过条件：

- `episodes: 16`；
- `frames: 14141`；
- `fps: 30`；
- `state_action_dim: 14`；
- 两个 wrist camera key；
- `act_smoke.loss` 是有限数；
- `finite_backward: true`。

任何一项失败都不要开始训练，先排查数据格式、视频解码、动作维度或依赖版本。

## 10. 运行 50-step GPU 冒烟训练

```bash
cd ~/output/code/project10_18
CUDA_VISIBLE_DEVICES=0 \
STEPS=50 \
BATCH_SIZE=2 \
NUM_WORKERS=0 \
bash scripts/train_overfit.sh
```

通过条件：

- 日志显示 `device: cuda`；
- `Effective batch size: 2 x 1 = 2`；
- loss 为有限值；
- 训练到 50/50；
- 日志最后出现 `End of training`；
- `~/output/runs/act_overfit_<时间>/checkpoints/000050` 存在。

50-step 只检查链路，不用其 loss 判断模型效果。

## 11. 后台运行 10k-step 五条数据过拟合

用后台方式运行，SSH 断开后训练仍会继续：

```bash
cd ~/output/code/project10_18
mkdir -p /tmp/triton_cache ~/output/runs

nohup env \
  CUDA_VISIBLE_DEVICES=0 \
  TRITON_CACHE_DIR=/tmp/triton_cache \
  STEPS=10000 \
  BATCH_SIZE=8 \
  NUM_WORKERS=8 \
  bash scripts/train_overfit.sh \
  > ~/output/runs/act_overfit_10000_console.log 2>&1 < /dev/null &

printf '%s\n' "$!" | tee ~/output/runs/act_overfit_10000.pid
```

`TRITON_CACHE_DIR=/tmp/triton_cache` 用于避免 DeepSpeed 把 Triton autotune cache 放到 NFS 后产生
警告或退出变慢。真正需要保留的模型不在 `/tmp`，而在 `~/output/runs`。

## 12. 监控训练

检查进程：

```bash
PID=$(cat ~/output/runs/act_overfit_10000.pid)
ps -p "$PID" -o pid,etime,stat,cmd
```

实时看日志：

```bash
tail -f ~/output/runs/act_overfit_10000_console.log
```

按 `Ctrl+C` 只会退出 `tail`，不会停止后台训练。

查看易读的 loss 记录：

```bash
tr '\r' '\n' < ~/output/runs/act_overfit_10000_console.log \
  | grep 'step:' \
  | tail -n 20
```

查看 GPU：

```bash
nvidia-smi
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
```

查找本次输出目录：

```bash
grep 'Output dir:' ~/output/runs/act_overfit_10000_console.log
```

查看 checkpoint：

```bash
find ~/output/runs/act_overfit_* -path '*/checkpoints/*' -maxdepth 3 -type d | sort
```

本次实测 GPU 显存约 5.2 GB、稳定速度约 16 steps/s。速度会随数据、worker 和硬件负载变化，
不要把这个数当作硬性标准。

## 13. 确认训练完成和模型完整

```bash
PID=$(cat ~/output/runs/act_overfit_10000.pid)
if ps -p "$PID" >/dev/null; then
  echo "training is still running"
else
  echo "training process exited"
fi

tail -n 20 ~/output/runs/act_overfit_10000_console.log
```

日志必须包含：

```text
Checkpoint policy after step 10000
End of training
```

假设日志显示输出目录为 `~/output/runs/act_overfit_YYYYMMDD_HHMMSS`，检查：

```bash
RUN_DIR=~/output/runs/act_overfit_YYYYMMDD_HHMMSS
du -sh "$RUN_DIR/checkpoints/010000"
find "$RUN_DIR/checkpoints/010000/pretrained_model" -maxdepth 1 -type f -ls
```

必须至少有：

```text
config.json
model.safetensors
policy_preprocessor.json
policy_postprocessor.json
train_config.json
```

## 14. 常见问题

### `Network is unreachable`，卡在 `hf auth login`

原因：实例到 `huggingface.co:443` 不通，不是 token 错误。按 `Ctrl+C` 退出，当前小数据改用
Mac `scp`；大数据走 DM3/OSS 资产。

### `rsync: command not found`

原因：远程镜像没有 rsync。小数据直接使用 `scp -r`，不要为了 100 MB 数据修改基础镜像。

### `nvidia-smi` 显示多张卡

ACT 脚本不会自动多卡。命令前始终加：

```bash
CUDA_VISIBLE_DEVICES=0
```

### torchvision 视频弃用警告

日志可能提示未来迁移到 TorchCodec。这是 warning，不是本次失败；PyAV 15.1.0 已成功解码本数据。
后续升级 LeRobot 时再统一迁移，当前三周项目中不要临时更换视频后端。

### pip 很慢

先确认使用了阿里云镜像。PyAV wheel 约 40 MB，本次下载数分钟属于网络问题。不要在下载期间反复
重建虚拟环境。安装一旦完成，`~/output/envs/act` 会持久保存。

### loss 降低是否代表可以部署

不代表。小数据过拟合只证明实现和数据可学习。正式部署前仍需：

1. 正式毛巾任务数据；
2. 完整 episode 级 train/eval split；
3. 离线动作范围检查；
4. ALOHA 左右臂、夹爪方向和单位核对；
5. 限速、限幅和急停；
6. 低速真机 rollout。

## 15. 训练结束后的资源处理

完成实验后，在 DM3 页面点击实例“停止”，释放 H20。不要误点“删除”。停止实例不会删除
`~/output` 中的代码、数据、环境和 checkpoint。

再次启动后执行：

```bash
source ~/output/envs/act/bin/activate
cd ~/output/code/project10_18
git pull --ff-only
```

然后检查数据与 checkpoint 仍在：

```bash
ls ~/output/data
ls ~/output/runs
```

## 16. 正式毛巾项目的下一步

当前 16 条“测试”数据只用于验证流水线。下一阶段不要直接在它上面做所谓正式模型：

1. 先采 20–40 条固定初态、成功的毛巾折叠演示；
2. 转换后重新运行第 9 节验证；
3. 用 5 条新数据重复第 10–13 节过拟合；
4. 扩充到 50–100 条；
5. 按完整 episode 划分 train/eval，不能随机拆帧；
6. 训练正式 ACT，保存数据版本、配置、loss 曲线和 checkpoint；
7. 完成动作安全检查后再部署真机。

每个阶段都采用“先小规模验证、通过后再扩大”的停止条件。这样出现问题时能区分是采集、转换、
模型、训练还是部署环节，而不是一次性把所有变量混在一起。
