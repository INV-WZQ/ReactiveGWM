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
  --trainable_models dit
```

`--game` 目前支持：

- `sf2`
- `sf3`

游戏 profile 会决定默认视频尺寸、按钮 schema、固定 prompt fallback 等数据处理行为。

### 全量 DiT 微调

```bash
accelerate launch --num_processes 8 --multi_gpu \
  -m ReactiveGWM_Code.training.bidirectional.train \
  --game sf2 \
  --dataset_base_path <dataset-root> \
  --dataset_metadata_path <metadata.csv> \
  --model_paths '<model-paths-json>' \
  --output_path <output-dir> \
  --trainable_models dit
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
  --trainable_filter cross_attn
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
  --trainable_filter_exclude cross_attn
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
  --lora_target_modules cross_attn,self_attn \
  --lora_rank 32
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
