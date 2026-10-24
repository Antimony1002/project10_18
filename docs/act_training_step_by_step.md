# ALOHA 双臂 ACT 训练逐步操作说明

本文按当前 DM3 实例、LeRobot 0.4.4 和本项目目录编写。命令分为“Mac 终端”和
“训练服务器”两类。除非特别说明，每个代码块可整块复制执行。

## 1. 从 Mac 登录训练服务器

Mac 终端：

```bash
ssh zhengkefan_qizu@101.132.150.51 -p 14424
```

登录后确认 GPU：

```bash
nvidia-smi
```

项目与环境位置：

```bash
cd /home/users/zhengkefan_qizu/output/code/project10_18
git status --short --branch
/home/users/zhengkefan_qizu/output/envs/act/bin/python -c "import torch, lerobot; print(torch.__version__, lerobot.__version__, torch.cuda.is_available())"
```

如果是全新实例，首次准备目录、代码和环境：

```bash
mkdir -p /home/users/zhengkefan_qizu/output/{code,data,envs,runs}
cd /home/users/zhengkefan_qizu/output/code
git clone https://github.com/Antimony1002/project10_18.git
cd project10_18
python3 -m venv --system-site-packages /home/users/zhengkefan_qizu/output/envs/act
/home/users/zhengkefan_qizu/output/envs/act/bin/python -m pip install --upgrade pip
/home/users/zhengkefan_qizu/output/envs/act/bin/python -m pip install "lerobot==0.4.4"
```

已有代码时更新：

```bash
cd /home/users/zhengkefan_qizu/output/code/project10_18
git pull --ff-only
```

## 2. 检查原始 50 集 LeRobot 数据

训练服务器：

```bash
ls /home/users/zhengkefan_qizu/output/data/i2rt_a_dual_act_20261024_empty_tray_50ep/meta/info.json
ls /home/users/zhengkefan_qizu/output/data/i2rt_a_dual_act_20261024_cup_tray_50ep/meta/info.json
```

检查 episode 和帧数：

```bash
jq '{total_episodes,total_frames,fps}' /home/users/zhengkefan_qizu/output/data/i2rt_a_dual_act_20261024_empty_tray_50ep/meta/info.json
jq '{total_episodes,total_frames,fps}' /home/users/zhengkefan_qizu/output/data/i2rt_a_dual_act_20261024_cup_tray_50ep/meta/info.json
```

预期：每项任务 50 episodes、30 FPS；空托盘 35,683 帧，纸杯托盘 36,415 帧。

## 3. 创建 40/10 episode 划分

固定验证 episode：`4,9,14,19,24,29,34,39,44,49`。其余 40 集训练。

训练服务器：

```bash
cd /home/users/zhengkefan_qizu/output/code/project10_18

/home/users/zhengkefan_qizu/output/envs/act/bin/python scripts/prepare_episode_split.py \
  /home/users/zhengkefan_qizu/output/data/i2rt_a_dual_act_20261024_empty_tray_50ep \
  /home/users/zhengkefan_qizu/output/data/i2rt_a_dual_act_20261024_empty_tray_40train_10val

/home/users/zhengkefan_qizu/output/envs/act/bin/python scripts/prepare_episode_split.py \
  /home/users/zhengkefan_qizu/output/data/i2rt_a_dual_act_20261024_cup_tray_50ep \
  /home/users/zhengkefan_qizu/output/data/i2rt_a_dual_act_20261024_cup_tray_40train_10val
```

输出目录已经存在时脚本会主动停止，不会覆盖。查看划分：

```bash
jq . /home/users/zhengkefan_qizu/output/data/i2rt_a_dual_act_20261024_empty_tray_40train_10val/meta/episode_split.json
jq . /home/users/zhengkefan_qizu/output/data/i2rt_a_dual_act_20261024_cup_tray_40train_10val/meta/episode_split.json
```

训练 episode 参数，两项任务相同：

```bash
TRAIN_EPISODES='[0,1,2,3,5,6,7,8,10,11,12,13,15,16,17,18,20,21,22,23,25,26,27,28,30,31,32,33,35,36,37,38,40,41,42,43,45,46,47,48]'
```

## 4. W&B 登录与检查

训练服务器首次登录：

```bash
/home/users/zhengkefan_qizu/output/envs/act/bin/python -m wandb login
```

按提示打开授权页或粘贴 API key。不要把 API key 写入代码、日志或聊天。

检查是否登录：

```bash
/home/users/zhengkefan_qizu/output/envs/act/bin/python - <<'PY'
import wandb
api = wandb.Api()
print("authenticated:", bool(api.api_key))
print("entity:", api.default_entity)
PY
```

当前 W&B 项目地址：

```text
https://wandb.ai/antimony1002-shanghai-jiao-tong-university/aloha-act-tray-validation
```

## 5. 正式训练命令

先确认 GPU 0、1 空闲：

```bash
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
pgrep -af lerobot_train || true
```

训练公共参数：10,000 steps、batch size 16、每 1,000 steps 保存、W&B 在线记录。

### 5.1 空托盘训练命令（GPU 0）

```bash
cd /home/users/zhengkefan_qizu/output/code/project10_18
TRAIN_EPISODES='[0,1,2,3,5,6,7,8,10,11,12,13,15,16,17,18,20,21,22,23,25,26,27,28,30,31,32,33,35,36,37,38,40,41,42,43,45,46,47,48]'
EMPTY_OUTPUT=/home/users/zhengkefan_qizu/output/runs/act_empty_tray_40train_10val_10k_$(date +%Y%m%d_%H%M%S)

nohup env \
  CUDA_VISIBLE_DEVICES=0 \
  TASK=empty_tray \
  DATASET_NAME=i2rt_a_dual_act_20261024_empty_tray_40train_10val \
  DATASET_ROOT=/home/users/zhengkefan_qizu/output/data/i2rt_a_dual_act_20261024_empty_tray_40train_10val \
  REPO_ID=Hydrogen1002/i2rt_a_dual_act_20261024_empty_tray_40train_10val \
  OUTPUT_DIR="$EMPTY_OUTPUT" \
  EPISODES="$TRAIN_EPISODES" \
  STEPS=10000 SAVE_FREQ=1000 BATCH_SIZE=16 NUM_WORKERS=8 \
  WANDB_ENABLE=true WANDB_MODE=online WANDB_DISABLE_ARTIFACT=true \
  WANDB_PROJECT=aloha-act-tray-validation \
  WANDB_ENTITY=antimony1002-shanghai-jiao-tong-university \
  bash scripts/train_act.sh >"${EMPTY_OUTPUT}.log" 2>&1 &

echo $! | tee "${EMPTY_OUTPUT}.pid"
echo "$EMPTY_OUTPUT"
```

### 5.2 纸杯托盘训练命令（GPU 1）

```bash
cd /home/users/zhengkefan_qizu/output/code/project10_18
TRAIN_EPISODES='[0,1,2,3,5,6,7,8,10,11,12,13,15,16,17,18,20,21,22,23,25,26,27,28,30,31,32,33,35,36,37,38,40,41,42,43,45,46,47,48]'
CUP_OUTPUT=/home/users/zhengkefan_qizu/output/runs/act_cup_tray_40train_10val_10k_$(date +%Y%m%d_%H%M%S)

nohup env \
  CUDA_VISIBLE_DEVICES=1 \
  TASK=cup_tray \
  DATASET_NAME=i2rt_a_dual_act_20261024_cup_tray_40train_10val \
  DATASET_ROOT=/home/users/zhengkefan_qizu/output/data/i2rt_a_dual_act_20261024_cup_tray_40train_10val \
  REPO_ID=Hydrogen1002/i2rt_a_dual_act_20261024_cup_tray_40train_10val \
  OUTPUT_DIR="$CUP_OUTPUT" \
  EPISODES="$TRAIN_EPISODES" \
  STEPS=10000 SAVE_FREQ=1000 BATCH_SIZE=16 NUM_WORKERS=8 \
  WANDB_ENABLE=true WANDB_MODE=online WANDB_DISABLE_ARTIFACT=true \
  WANDB_PROJECT=aloha-act-tray-validation \
  WANDB_ENTITY=antimony1002-shanghai-jiao-tong-university \
  bash scripts/train_act.sh >"${CUP_OUTPUT}.log" 2>&1 &

echo $! | tee "${CUP_OUTPUT}.pid"
echo "$CUP_OUTPUT"
```

## 6. 日志查看命令

当前这次训练的日志：

```bash
tail -F \
  /home/users/zhengkefan_qizu/output/runs/act_empty_tray_40train_10val_10k_20261024_230628.log \
  /home/users/zhengkefan_qizu/output/runs/act_cup_tray_40train_10val_10k_20261024_230628.log
```

按 `Ctrl+C` 只退出日志查看，不会停止后台训练。

查看 GPU：

```bash
watch -n 2 nvidia-smi
```

查看训练进程：

```bash
pgrep -af lerobot_train
```

查看最近的 step/loss：

```bash
grep -o 'step:[^\r]*' /home/users/zhengkefan_qizu/output/runs/act_empty_tray_40train_10val_10k_20261024_230628.log | tail
grep -o 'step:[^\r]*' /home/users/zhengkefan_qizu/output/runs/act_cup_tray_40train_10val_10k_20261024_230628.log | tail
```

确认训练完成：

```bash
grep 'End of training' /home/users/zhengkefan_qizu/output/runs/act_*_40train_10val_10k_20261024_230628.log
```

## 7. checkpoint 检查命令

```bash
find /home/users/zhengkefan_qizu/output/runs/act_empty_tray_40train_10val_10k_20261024_230628/checkpoints -mindepth 1 -maxdepth 1 -type d | sort
find /home/users/zhengkefan_qizu/output/runs/act_cup_tray_40train_10val_10k_20261024_230628/checkpoints -mindepth 1 -maxdepth 1 -type d | sort
```

部署模型位于：

```text
checkpoints/001000/pretrained_model/
checkpoints/002000/pretrained_model/
...
checkpoints/010000/pretrained_model/
```

`training_state/` 用于断点续训；普通部署只需要 `pretrained_model/`。

## 8. 从服务器下载全部模型 checkpoint

以下命令在 Mac 终端运行。先设置服务器 run 和本地目录：

```bash
REMOTE_EMPTY_RUN=/home/users/zhengkefan_qizu/output/runs/act_empty_tray_40train_10val_10k_20261024_230628
REMOTE_CUP_RUN=/home/users/zhengkefan_qizu/output/runs/act_cup_tray_40train_10val_10k_20261024_230628
LOCAL_CHECKPOINTS=/Users/donglujia/lerobot/project/outputs/act_validation_checkpoints_20261024
mkdir -p "$LOCAL_CHECKPOINTS"/{empty_tray,cup_tray}
```

下载 1k–10k 全部可部署模型：

```bash
for STEP in 001000 002000 003000 004000 005000 006000 007000 008000 009000 010000; do
  mkdir -p "$LOCAL_CHECKPOINTS/empty_tray/$STEP" "$LOCAL_CHECKPOINTS/cup_tray/$STEP"
  scp -q -P 14424 -r \
    "zhengkefan_qizu@101.132.150.51:${REMOTE_EMPTY_RUN}/checkpoints/${STEP}/pretrained_model" \
    "$LOCAL_CHECKPOINTS/empty_tray/$STEP/"
  scp -q -P 14424 -r \
    "zhengkefan_qizu@101.132.150.51:${REMOTE_CUP_RUN}/checkpoints/${STEP}/pretrained_model" \
    "$LOCAL_CHECKPOINTS/cup_tray/$STEP/"
done
```

核对数量：

```bash
find "$LOCAL_CHECKPOINTS" -name model.safetensors -type f | sort
find "$LOCAL_CHECKPOINTS" -name model.safetensors -type f | wc -l
du -sh "$LOCAL_CHECKPOINTS"
```

两项任务各 10 个时总数应为 20。

## 9. Hugging Face 登录和推送命令

Mac 首次登录：

```bash
HF_ENDPOINT=https://huggingface.co hf auth login
HF_ENDPOINT=https://huggingface.co hf auth whoami
```

创建私有模型仓库：

```bash
HF_ENDPOINT=https://huggingface.co hf repos create \
  Hydrogen1002/act-i2rt-empty-tray-40train-10val --type model --private --exist-ok

HF_ENDPOINT=https://huggingface.co hf repos create \
  Hydrogen1002/act-i2rt-cup-tray-40train-10val --type model --private --exist-ok
```

上传全部 checkpoint：

```bash
HF_ENDPOINT=https://huggingface.co hf upload \
  Hydrogen1002/act-i2rt-empty-tray-40train-10val \
  "$LOCAL_CHECKPOINTS/empty_tray" . --repo-type model \
  --commit-message "Add ACT 40-train checkpoints 1k-10k"

HF_ENDPOINT=https://huggingface.co hf upload \
  Hydrogen1002/act-i2rt-cup-tray-40train-10val \
  "$LOCAL_CHECKPOINTS/cup_tray" . --repo-type model \
  --commit-message "Add ACT 40-train checkpoints 1k-10k"
```

## 10. 从 Hugging Face 下载模型

Mac 或其他已登录机器：

```bash
HF_ENDPOINT=https://huggingface.co hf download \
  Hydrogen1002/act-i2rt-empty-tray-40train-10val \
  --local-dir ./act-empty-tray-40train-10val

HF_ENDPOINT=https://huggingface.co hf download \
  Hydrogen1002/act-i2rt-cup-tray-40train-10val \
  --local-dir ./act-cup-tray-40train-10val
```

只下载某一个 checkpoint，例如 4k：

```bash
HF_ENDPOINT=https://huggingface.co hf download \
  Hydrogen1002/act-i2rt-empty-tray-40train-10val \
  --include '004000/pretrained_model/**' \
  --local-dir ./act-empty-tray-4k
```

## 11. 停止错误训练

仅在参数或数据确实错误时执行。先查 PID：

```bash
pgrep -af lerobot_train
```

先正常终止指定父进程：

```bash
kill -TERM <父进程PID>
```

等待几秒后确认：

```bash
pgrep -af lerobot_train || true
nvidia-smi
```

不要使用 `kill -9`、模糊的 `pkill python` 或删除整个 runs 目录。

## 12. 当前训练记录

```text
空托盘输出：/home/users/zhengkefan_qizu/output/runs/act_empty_tray_40train_10val_10k_20261024_230628
纸杯托盘输出：/home/users/zhengkefan_qizu/output/runs/act_cup_tray_40train_10val_10k_20261024_230628
空托盘父 PID：1262423
纸杯托盘父 PID：1262424
W&B project：aloha-act-tray-validation
```
