#!/usr/bin/env python3
"""Resume the cup-placement PI0 run at step 3000 through step 5000."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        from train_pi0_cup_place_then_carry import worker

        worker()
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--save-freq", type=int, default=500)
    parser.add_argument("--cuda-visible-devices", default="1,2,3")
    parser.add_argument("--num-processes", type=int, default=3)
    parser.add_argument("--wandb-project", default="i2rt_pi0_cup_place_then_carry")
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    parser.add_argument("--dry-run", action="store_true")
    options = parser.parse_args()

    checkpoint = options.checkpoint.resolve()
    config_path = checkpoint / "pretrained_model/train_config.json"
    step_path = checkpoint / "training_state/training_step.json"
    if not config_path.is_file() or not step_path.is_file():
        raise FileNotFoundError("Checkpoint is missing its model config or training state")
    start_step = int(json.loads(step_path.read_text())["step"])
    if start_step != 3000 or options.steps <= start_step:
        raise ValueError(f"Expected a 3000-step checkpoint and target beyond it; got {start_step}, {options.steps}")
    if options.num_processes != len(options.cuda_visible_devices.split(",")):
        raise ValueError("Process count must match selected GPUs")
    if "0" in options.cuda_visible_devices.split(","):
        raise ValueError("GPU 0 is reserved")

    run_root = Path.home() / "output/runs/project10_18_pi0"
    run = run_root / ("resume_pi0_3000_to_" + str(options.steps) + "_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    wandb_run_id = "p0" + secrets.token_hex(3)
    overrides = [
        f"--config_path={config_path}",
        "--resume=true",
        f"--steps={options.steps}",
        f"--output_dir={run}",
        f"--save_freq={options.save_freq}",
        "--save_checkpoint=true",
        "--eval_freq=0",
        "--log_freq=50",
        "--wandb.enable=true",
        f"--wandb.project={options.wandb_project}",
        f"--wandb.mode={options.wandb_mode}",
        f"--wandb.run_id={wandb_run_id}",
        "--wandb.disable_artifact=true",
        # The first 3000-step run auto-scaled its cosine schedule to decay at step 3000.
        # Keep the same 2.5e-6 decay floor instead of jumping to a new LR curve.
        "--scheduler.num_warmup_steps=100",
        "--scheduler.num_decay_steps=3000",
    ]
    if options.dry_run:
        from lerobot.configs.train import TrainPipelineConfig
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy  # registers pi0 config

        del PI0Policy
        sys.argv = [sys.argv[0], *overrides]
        cfg = TrainPipelineConfig.from_pretrained(config_path, cli_args=overrides[1:])
        cfg.validate()
        if not cfg.resume or cfg.checkpoint_path.resolve() != checkpoint or cfg.steps != options.steps:
            raise RuntimeError("Resume config did not select the requested checkpoint and target step")
        print("Checkpoint:", checkpoint)
        print("Source step:", start_step)
        print("Target step:", options.steps)
        print("GPUs:", options.cuda_visible_devices)
        print("W&B mode:", options.wandb_mode)
        print("Output:", run)
        print("Overrides:", overrides)
        return

    if shutil.disk_usage(run_root).free < 200 * 1024**3:
        raise RuntimeError("Less than 200 GiB free in output")
    run_root.mkdir(parents=True, exist_ok=True)
    log = run_root / (run.name + ".log")
    args = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        *(["--multi_gpu"] if options.num_processes > 1 else []),
        f"--num_processes={options.num_processes}",
        "--num_machines=1",
        "--mixed_precision=bf16",
        "--dynamo_backend=no",
        str(Path(__file__).resolve()),
        "--worker",
        *overrides,
    ]
    env = dict(
        os.environ,
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        CUDA_VISIBLE_DEVICES=options.cuda_visible_devices,
        I2RT_PI0_BASE_DIR=str(Path.home() / "output/models/huang_pi0/pi0_base"),
        WANDB_DIR=str(run_root),
    )
    print("Log:", log, flush=True)
    with log.open("x") as stream:
        with subprocess.Popen(args, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                stream.write(line)
                stream.flush()
            code = process.wait()
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
