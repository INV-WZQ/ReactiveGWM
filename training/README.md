# ReactiveGWM 训练说明（阶段一）

本目录目前只包含阶段一：普通双向训练（bidirectional training）。Causal Forcing 训练尚未迁移到这里。

## 依赖

训练代码继续依赖本仓库同级的 `DiffSynth-Studio/diffsynth`。建议在独立环境中安装：

```bash
pip install -r ReactiveGWM_Code/requirements.txt
pip install -r ReactiveGWM_Code/training/requirements.txt
pip install -e DiffSynth-Studio
```

如果不执行 editable install，也可以通过 `PYTHONPATH` 使用本地源码：

```bash
export PYTHONPATH=/path/to/ReactiveGWM/DiffSynth-Studio:/path/to/ReactiveGWM:${PYTHONPATH}
```

## 验证入口

```bash
python -m compileall ReactiveGWM_Code/training/data ReactiveGWM_Code/training/bidirectional
python -m ReactiveGWM_Code.training.bidirectional.train --help
python -m ReactiveGWM_Code.training.bidirectional.precompute_cache --help
```

也支持从 `ReactiveGWM_Code/` 目录内直接运行脚本：

```bash
python training/bidirectional/train.py --help
python training/bidirectional/precompute_cache.py --help
```

## 普通双向训练

入口：

```bash
python -m ReactiveGWM_Code.training.bidirectional.train
```

最小命令形态：

```bash
accelerate launch --num_processes 8 --multi_gpu \
  -m ReactiveGWM_Code.training.bidirectional.train \
  --game sf2 \
  --dataset_base_path <dataset-root> \
  --dataset_metadata_path <metadata.csv> \
  --model_paths '<model-paths-json>' \
  --output_path <output-dir> \
  --trainable_models dit \
  --learning_rate 5e-5 \
  --max_train_steps 30000 \
  --save_steps 1000
```

`--game` 目前支持：

- `sf2`
- `sf3`

游戏 profile 会决定默认视频尺寸、按钮 schema、固定 prompt fallback 等数据处理行为。

### 训练步数和超参数

训练入口复用 DiffSynth 的通用训练参数，并额外支持 `--max_train_steps` 精确限制训练步数。建议显式传入训练超参数，不依赖默认值：

| 参数 | 作用 | 推荐值 / 说明 |
|---|---|---|
| `--learning_rate` | AdamW 学习率 | Full / scoped DiT: `5e-5`；LoRA: `1e-4` |
| `--weight_decay` | AdamW weight decay | `0.01` |
| `--num_epochs` | 最大 epoch 数 | 不使用 `--max_train_steps` 时控制总训练轮数；常用 `1000` |
| `--max_train_steps` | 精确训练多少 step 后停止 | 推荐按目标 checkpoint step 显式设置，例如 `30000` |
| `--save_steps` | 每多少 step 保存一次 checkpoint | `1000` |
| `--gradient_accumulation_steps` | 梯度累积步数 | Full DiT: `1`；scoped / LoRA: `2` |
| `--dataset_repeat` | 每个 epoch 重复数据集次数 | `1` |
| `--dataset_num_workers` | DataLoader worker 数 | `4` |
| `--height --width --num_frames` | 视频尺寸和帧数 | SF2: `480 608 101`；SF3: `480 832 101` |
| `--use_gradient_checkpointing` | 降低显存占用 | 推荐开启 |
| `--prompt_dropout_prob` | prompt CFG-style dropout | `0.1` |
| `--action_dropout_prob` | action dropout | `0.0` |

这里的 step 与 `--save_steps`、`--max_train_steps`、checkpoint 文件名 `step-N.safetensors` 使用同一计数：prepared dataloader 每迭代一次计 1 step。当前 DataLoader batch size 固定为 1；在多卡训练时：

```text
steps_per_epoch ~= ceil(num_samples * dataset_repeat / num_processes)
total_steps     = num_epochs * steps_per_epoch              # without --max_train_steps
total_steps     = min(num_epochs * steps_per_epoch, N)       # with --max_train_steps N
effective_batch = num_processes * gradient_accumulation_steps
```

`--gradient_accumulation_steps` 改变等效 batch size 和 optimizer 更新频率，但不改变 `save_steps` / `max_train_steps` 的计数单位。若没有设置 `--max_train_steps`，训练会按 `--num_epochs` 跑完；若设置了 `--max_train_steps`，达到该 step 后会停止并保存最终 checkpoint。

推荐从下面几组配置开始，再按数据规模和显存调整。

### 全量 DiT 微调

```bash
accelerate launch --num_processes 8 --multi_gpu \
  -m ReactiveGWM_Code.training.bidirectional.train \
  --game sf2 \
  --dataset_base_path <dataset-root> \
  --dataset_metadata_path <metadata.csv> \
  --model_paths '<model-paths-json>' \
  --output_path <output-dir> \
  --trainable_models dit \
  --learning_rate 5e-5 \
  --max_train_steps 30000 \
  --save_steps 1000 \
  --gradient_accumulation_steps 1 \
  --weight_decay 0.01 \
  --dataset_repeat 1 \
  --dataset_num_workers 4 \
  --use_gradient_checkpointing \
  --prompt_dropout_prob 0.1 \
  --action_dropout_prob 0.0
```

### 指定模块微调

只训练名称包含某些子串的 DiT 参数：

```bash
accelerate launch --num_processes 8 --multi_gpu \
  -m ReactiveGWM_Code.training.bidirectional.train \
  --game sf2 \
  --dataset_base_path <dataset-root> \
  --dataset_metadata_path <metadata.csv> \
  --model_paths '<model-paths-json>' \
  --output_path <output-dir> \
  --trainable_models dit \
  --trainable_filter cross_attn \
  --learning_rate 5e-5 \
  --max_train_steps 30000 \
  --save_steps 1000 \
  --gradient_accumulation_steps 2 \
  --weight_decay 0.01 \
  --dataset_repeat 1 \
  --dataset_num_workers 4 \
  --use_gradient_checkpointing \
  --prompt_dropout_prob 0.1 \
  --action_dropout_prob 0.0
```

排除名称包含某些子串的 DiT 参数：

```bash
accelerate launch --num_processes 8 --multi_gpu \
  -m ReactiveGWM_Code.training.bidirectional.train \
  --game sf2 \
  --dataset_base_path <dataset-root> \
  --dataset_metadata_path <metadata.csv> \
  --model_paths '<model-paths-json>' \
  --output_path <output-dir> \
  --trainable_models dit \
  --trainable_filter_exclude cross_attn \
  --learning_rate 5e-5 \
  --max_train_steps 30000 \
  --save_steps 1000 \
  --gradient_accumulation_steps 2 \
  --weight_decay 0.01 \
  --dataset_repeat 1 \
  --dataset_num_workers 4 \
  --use_gradient_checkpointing \
  --prompt_dropout_prob 0.1 \
  --action_dropout_prob 0.0
```

### LoRA 微调

```bash
accelerate launch --num_processes 8 --multi_gpu \
  -m ReactiveGWM_Code.training.bidirectional.train \
  --game sf2 \
  --dataset_base_path <dataset-root> \
  --dataset_metadata_path <metadata.csv> \
  --model_paths '<model-paths-json>' \
  --output_path <output-dir> \
  --lora_base_model dit \
  --lora_target_modules q,k,v,o \
  --lora_rank 32 \
  --learning_rate 1e-4 \
  --max_train_steps 30000 \
  --save_steps 1000 \
  --gradient_accumulation_steps 2 \
  --weight_decay 0.01 \
  --dataset_repeat 1 \
  --dataset_num_workers 4 \
  --use_gradient_checkpointing \
  --prompt_dropout_prob 0.1 \
  --action_dropout_prob 0.0
```

## Cache 预计算

入口：

```bash
python -m ReactiveGWM_Code.training.bidirectional.precompute_cache
```

示例：

```bash
python -m ReactiveGWM_Code.training.bidirectional.precompute_cache \
  --game sf2 \
  --metadata_path <metadata.csv> \
  --dataset_base_path <dataset-root> \
  --cache_root <cache-root> \
  --model_paths '<model-paths-json>' \
  --tokenizer_path <umt5-dir> \
  --height 480 \
  --width 608 \
  --num_frames 101
```

预计算完成后，训练时加上：

```bash
--use_cached_dataset --cache_root <cache-root>
```

`precompute_cache.py` 和 `bidirectional/cached_dataset.py` 的 cache key 逻辑必须保持一致；修改其中一个时需要同步检查另一个。

## Resume

阶段一训练入口支持两类恢复方式：

- `--resume_from_ckpt <step.safetensors>`：加载已保存的 DiT 权重，不恢复 optimizer / LR / RNG / dataloader 状态。
- `--save_full_state` + `--resume_state <state-dir>`：使用 `accelerator.save_state/load_state` 恢复完整训练状态。

二者互斥。

当 `--resume_state` 指向 `state-N` 目录时，训练 step 计数会从 `N` 继续；因此 `--max_train_steps 30000` 表示训练到全局 step 30000，而不是在 resume 后再额外训练 30000 step。
