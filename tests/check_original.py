"""Compare the supplied v2 training source with this release on identical inputs."""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
import torch

from fixtures import TRAINING, SMALL_MODEL, batch, text_tensors, forward


def worker(args):
    if args.original_training:
        source = Path(args.original_training).resolve()
        sys.path[:0] = [str(source / 'src'), str(source)]
        from common.config import load_config, make_model, bind_text, original_optimizer_config
        from common.optim import build_training_optimizer, build_training_scheduler
    else:
        from selective_agency.runtime.config import load_config, make_model, bind_text, original_optimizer_config
        from selective_agency.runtime.optim import build_training_optimizer, build_training_scheduler
    from selective_agency.loss import WanFlowMatchingObjective
    torch.set_num_threads(1)
    config = load_config(TRAINING / 'training/configs' / ('sf3.yaml' if args.game == 'sf' else 'hnm.yaml'))
    config['model'].update(SMALL_MODEL)
    config['training']['precision'] = args.precision
    torch.manual_seed(17)
    model = make_model(config, args.device, trainable=True)
    bind_text(model, text_tensors())
    dtype = next(model.parameters()).dtype
    value = {k: v.to(args.device, dtype=dtype if k in {'input_latents', 'first_frame_latents'} else v.dtype)
             for k, v in batch(dtype).items()}
    optimizer = build_training_optimizer(model, original_optimizer_config(config))
    scheduler = build_training_scheduler(optimizer, original_optimizer_config(config))
    objective = WanFlowMatchingObjective()
    result = {'forward': forward(model, value).detach()}
    for update in range(2):
        loss = objective(model, value, use_gradient_checkpointing=True, timestep_id=234 + update,
                         noise=torch.ones_like(value['input_latents']) * (update + .5))
        loss.loss.backward()
        result[f'loss.{update}'] = loss.loss.detach()
        if update == 0:
            result.update({'grad.' + name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None})
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    result.update({'weight.' + name: value for name, value in model.state_dict().items()})
    torch.save(result, args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--game', choices=('hnm', 'sf'), required=True)
    parser.add_argument('--original-training', help='training/ directory extracted from the supplied zip')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--precision', choices=('fp32', 'bf16'), default='fp32')
    parser.add_argument('--output', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.output:
        worker(args)
        return
    if not args.original_training:
        parser.error('provide --original-training for a numerical comparison')
    with tempfile.TemporaryDirectory(prefix='reactivegwm-parity-') as temporary:
        outputs = []
        for name in ['original', 'release']:
            output = Path(temporary) / (name + '.pt')
            command = [sys.executable, __file__, '--game', args.game, '--device', args.device,
                       '--precision', args.precision, '--output', str(output)]
            if name == 'original':
                command += ['--original-training', args.original_training]
            subprocess.run(command, check=True)
            outputs.append(torch.load(output, weights_only=True, map_location='cpu'))
        a, b = outputs
        assert a.keys() == b.keys()
        for key in a:
            torch.testing.assert_close(a[key], b[key], atol=0, rtol=0, msg=lambda msg: f'{key}: {msg}')
        print(f'PASS {args.game} {args.precision} {args.device}: {len(a)} forward/loss/gradient/update tensors exactly equal')


if __name__ == '__main__':
    main()
