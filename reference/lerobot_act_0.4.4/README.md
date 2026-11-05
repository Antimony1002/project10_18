# ACT 训练源码阅读副本

这里保存的是 DM3 实例中 **LeRobot 0.4.4** 环境实际安装的源码，供阅读、批注和复盘使用。三个 Python 文件按原样复制，未修改。项目自己的训练启动脚本仍在 [`../../scripts/train_act.sh`](../../scripts/train_act.sh)；该脚本与实例中的同名文件 SHA-256 一致。

## 从哪里读起

1. [`configuration_act.py`](configuration_act.py)：`ACTConfig` 定义输入、动作块、网络层数、VAE、损失权重和优化器默认值。
2. [`modeling_act.py`](modeling_act.py)：`ACTPolicy` 从第 41 行开始。先看 `forward()`（训练损失，约第 136 行）和 `select_action()`（推理，约第 100 行），再看 `ACT`（约第 256 行）、`ACTEncoder`（约第 513 行）和 `ACTDecoder`（约第 571 行）。
3. [`lerobot_train.py`](lerobot_train.py)：LeRobot 通用训练循环；它加载数据与 policy、执行优化、记录日志并保存 checkpoint。
4. [`../../scripts/train_act.sh`](../../scripts/train_act.sh)：我们自己的命令行参数包装层。它预设的是早期两个托盘任务，**不能用其默认任务名推断任务三的配置**。

任务三的准确参数保存在实例：`/home/users/zhengkefan_qizu/output/runs/project10_18_act/cup_place_then_carry_10k_20261104/checkpoints/010000/pretrained_model/train_config.json`。本次使用两路腕部 RGB、14 维状态与 14 维动作、60 步动作块、ResNet18、`dim_model=512`、8 个注意力头、4 层 encoder、1 层 decoder、VAE latent 32、`kl_weight=10`、学习率 `1e-5`、batch 16、10,000 步、每 1,000 步保存。

## 来源与校验

实例：`ssh zhengkefan_qizu@101.132.150.51 -p 14868`；安装目录：`/home/users/zhengkefan_qizu/output/envs/act/lib/python3.12/site-packages/lerobot/`。

| 本地文件 | 实例中的相对路径 | SHA-256 |
| --- | --- | --- |
| `modeling_act.py` | `policies/act/modeling_act.py` | `69301fff11cbb8f4ba7bfe473a55ba488c3d2f89c25ecdfe512347a93b4ad04f` |
| `configuration_act.py` | `policies/act/configuration_act.py` | `11aa9037f981a957d63632561c795fac243e7898ee9302f7afac2f5d45920f7b` |
| `lerobot_train.py` | `scripts/lerobot_train.py` | `827a2b7e74341f1d89d5c508d62ce52592bb9cab0c74fe8a3de38e9f03886410` |

这些文件是阅读副本，训练仍调用实例环境中安装的 LeRobot。修改副本不会改变实例中的模型或训练结果。上游版权声明保留在源文件中；许可证见 [`LICENSE`](LICENSE)（Apache-2.0）。
