"""Fine-tune LeRobot 0.4.4 PI0 on the separate cup-placement tray dataset.

The strict checkpoint alias loader follows the lab's proven PI0 training entry.
This script never edits the shared base checkpoint or the teammate's run.
"""

import shutil
import json
import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def remove_disabled_steps(payload):
    steps = []
    for step in payload["steps"]:
        if step.get("registry_name") in {
            "relative_actions_processor", "absolute_actions_processor"
        }:
            if step.get("config", {}).get("enabled") is not False:
                raise RuntimeError("An action transform is enabled; do not remove it.")
        else:
            steps.append(step)
    return {**payload, "steps": steps}


def install_alias_patch(policy_class, aliases):
    original_load = policy_class.load_state_dict
    original_from = policy_class.from_pretrained.__func__

    def load(self, state_dict, strict=True, **kwargs):
        state_dict = state_dict.copy()
        expected = self.state_dict()
        for alias, source in aliases.items():
            alias = alias if alias.startswith("model.") else "model." + alias
            source = source if source.startswith("model.") else "model." + source
            if alias not in state_dict and alias in expected:
                if source not in state_dict:
                    raise RuntimeError("Missing shared weight source: " + source)
                if state_dict[source].shape != expected[alias].shape:
                    raise RuntimeError("Shared weight shape mismatch: " + alias)
                state_dict[alias] = state_dict[source]
        result = original_load(self, state_dict, strict=True, **kwargs)
        self._pi0_weights_verified = True
        return result

    def from_pretrained(cls, *args, **kwargs):
        model = original_from(cls, *args, **kwargs)
        if not getattr(model, "_pi0_weights_verified", False):
            raise RuntimeError("Pretrained weights were not strictly loaded; abort training.")
        print("PI0_WEIGHTS_STRICTLY_VERIFIED", flush=True)
        return model

    policy_class.load_state_dict = load
    policy_class.from_pretrained = classmethod(from_pretrained)


def worker():
    from safetensors import safe_open
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy

    weights = Path(os.environ["I2RT_PI0_BASE_DIR"]) / "model.safetensors"
    with safe_open(str(weights), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        keys = set(handle.keys())
    aliases = {key: value for key, value in metadata.items() if value in keys}
    alias = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    source = "paligemma_with_expert.paligemma.lm_head.weight"
    if aliases.get(alias) != source:
        raise RuntimeError("The shared embedding metadata differs from the inspected checkpoint.")
    install_alias_patch(PI0Policy, aliases)
    sys.argv = [sys.argv[0], *sys.argv[2:]]
    from lerobot.scripts.lerobot_train import main
    main()


def launch():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--save-freq", type=int, default=1000)
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--num-processes", type=int, default=4)
    parser.add_argument("--cuda-visible-devices", default="0,1,2,3")
    parser.add_argument("--dry-run", action="store_true")
    options = parser.parse_args()
    if options.steps < 1 or options.save_freq < 1 or options.num_processes < 1:
        parser.error("Steps, save frequency, and process count must be positive.")
    home = Path.home()
    root = home / "output/models/huang_pi0"
    base = root / "pi0_base"
    tokenizer = root / "pi0_tokenizer"
    prepared = home / "output/models/project10_18_pi0/pi0_training_base"
    dataset_root = home / "output/data/project10_18_pi0/i2rt_cup_place_then_carry_20261101_0116_0130"
    split_path = Path(__file__).resolve().parents[1] / "configs/pi0_cup_place_then_carry_split.json"
    split = json.loads(split_path.read_text())
    train_episodes, val_episodes = split["train_episodes"], split["val_episodes"]
    if len(train_episodes) != 12 or len(val_episodes) != 3:
        raise RuntimeError("Expected a 12/3 episode split for the 15-episode pilot.")
    if sorted(train_episodes + val_episodes) != list(range(15)):
        raise RuntimeError("Train/validation episodes must partition 0..14 exactly.")
    info = json.loads((dataset_root / "meta/info.json").read_text())
    if info["total_episodes"] != 15 or info["total_frames"] != 16769 or info["fps"] != 30:
        raise RuntimeError("Unexpected dataset version or frame count.")
    features = info["features"]
    expected = {"observation.images.left_wrist", "observation.images.right_wrist"}
    if {key for key in features if key.startswith("observation.images.")} != expected:
        raise RuntimeError("Expected exactly the two real wrist cameras.")
    if features["observation.state"]["shape"] != [14] or features["action"]["shape"] != [14]:
        raise RuntimeError("Expected 14D state and action.")
    if options.dry_run:
        print("Dataset:", dataset_root)
        print("Shared base (read only):", base)
        print("Tokenizer:", tokenizer)
        print("Prepared model (ours):", prepared)
        print("Train episodes:", train_episodes)
        print("Validation episodes:", val_episodes)
        print("No model or GPU was started.")
        return
    prepared.mkdir(parents=True, exist_ok=True)
    model_config = json.loads((base / "config.json").read_text())
    model_config["input_features"] = {}
    (prepared / "config.json").write_text(json.dumps(model_config, indent=2))
    weights_link = prepared / "model.safetensors"
    if not weights_link.exists():
        weights_link.symlink_to(base / "model.safetensors")
    elif weights_link.resolve() != (base / "model.safetensors").resolve():
        raise RuntimeError("Prepared weights do not point to the shared base checkpoint.")
    # Always rebuild processors from the untouched downloaded JSON files.
    for name in ["policy_preprocessor.json", "policy_postprocessor.json"]:
        payload = json.loads((base / name).read_text())
        payload = remove_disabled_steps(payload)
        for step in payload["steps"]:
            if step.get("registry_name") == "tokenizer_processor":
                step["config"]["tokenizer_name"] = str(tokenizer)
        (prepared / name).write_text(json.dumps(payload, indent=2))
    prefix = "train_4gpu_" if options.save_checkpoints else "smoke_fixed_"
    run = home / "output/runs/project10_18_pi0" / (prefix + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    run.parent.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(run.parent).free / 1024**3
    print(f"Available storage: {free_gib:.1f} GiB", flush=True)
    if options.save_checkpoints and free_gib < 200:
        raise RuntimeError("Less than 200 GiB available; check storage before saving large checkpoints.")
    args = [
        sys.executable, "-m", "accelerate.commands.launch",
        *(["--multi_gpu"] if options.num_processes > 1 else []),
        f"--num_processes={options.num_processes}", "--num_machines=1", "--mixed_precision=bf16",
        "--dynamo_backend=no", str(Path(__file__).resolve()), "--worker",
        f"--policy.path={prepared}", "--policy.device=cuda",
        "--policy.dtype=bfloat16", "--policy.gradient_checkpointing=true",
        "--policy.compile_model=false", "--policy.train_expert_only=false",
        "--policy.freeze_vision_encoder=false", "--policy.push_to_hub=false",
        "--dataset.repo_id=Hydrogen1002/i2rt_cup_place_then_carry_lerobot_20261101_0116_0130",
        f"--dataset.root={dataset_root}",
        "--dataset.episodes=" + json.dumps(train_episodes),
        "--dataset.video_backend=pyav", f"--output_dir={run}",
        "--job_name=project10_18_pi0", "--batch_size=2", "--num_workers=0",
        f"--steps={options.steps}", "--log_freq=50" if options.save_checkpoints else "--log_freq=5",
        "--eval_freq=0", f"--save_freq={options.save_freq}",
        "--save_checkpoint=" + str(options.save_checkpoints).lower(),
        "--wandb.enable=false", "--seed=42",
    ]
    env = dict(os.environ, HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
               CUDA_VISIBLE_DEVICES=options.cuda_visible_devices, I2RT_PI0_BASE_DIR=str(base))
    log = run.parent / (run.name + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
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
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker()
    else:
        launch()
