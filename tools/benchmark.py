import argparse
import time

import torch
# Delay importing mmcv/mmengine/mmseg until runtime so --help and CLI work even when
# the installed packages expose different top-level APIs.



def parse_args():
    parser = argparse.ArgumentParser(description='MMSeg benchmark a model')
    parser.add_argument('config', help='test config file path')
    # parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--log-interval', type=int, default=50, help='interval of logging')
    parser.add_argument(
        '--samples-per-gpu', type=int, default=1, help='images per GPU (batch size per device)')
    parser.add_argument(
        '--total-images', type=int, default=200, help='total number of images to benchmark (across all GPUs)')
    parser.add_argument(
        '--num-warmup', type=int, default=5, help='number of warmup iterations to skip when measuring')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # Adaptive imports: prefer mmengine (newer unified API), fall back to mmcv.
    Config = None
    MMDataParallel = None
    load_checkpoint = None
    build_dataset = None
    build_dataloader = None
    build_segmentor = None

    try:
        import mmengine
        Config = mmengine.Config
        try:
            from mmengine.model import MMDataParallel as _MMDataParallel
            MMDataParallel = _MMDataParallel
        except Exception:
            MMDataParallel = None
        try:
            from mmengine.runner import load_checkpoint as _load_ckpt
            load_checkpoint = _load_ckpt
        except Exception:
            load_checkpoint = None
    except Exception:
        # mmengine not available; try mmcv
        try:
            from mmcv import Config as _Config
            Config = _Config
        except Exception:
            Config = None
        try:
            from mmcv.parallel import MMDataParallel as _MMDataParallel
            MMDataParallel = _MMDataParallel
        except Exception:
            MMDataParallel = None
        try:
            from mmcv.runner import load_checkpoint as _load_ckpt
            load_checkpoint = _load_ckpt
        except Exception:
            load_checkpoint = None

    # Import mmseg helpers (will raise if mmseg isn't installed)
    try:
        from mmseg.datasets import build_dataloader as _build_dataloader, build_dataset as _build_dataset
        from mmseg.models import build_segmentor as _build_segmentor
        build_dataset = _build_dataset
        build_dataloader = _build_dataloader
        build_segmentor = _build_segmentor
    except Exception as e:
        print('Required mmseg imports failed:', e)
        print('Install a compatible mmseg/mmcv/mmengine combo or run without benchmarking.')
        return

    if Config is None:
        print('Neither mmengine.Config nor mmcv.Config is importable in this environment.')
        print('Install mmengine or mmcv compatible with your setup.')
        return

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    torch.backends.cudnn.benchmark = False
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=args.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    # load_checkpoint(model, args.checkpoint, map_location='cpu')

    model = MMDataParallel(model, device_ids=[0])

    model.eval()

    # the first several iterations may be very slow so skip them
    num_warmup = args.num_warmup
    pure_inf_time = 0.0
    images_processed = 0
    total_images = args.total_images

    # benchmark and take the average across images
    for i, data in enumerate(data_loader):

        # determine how many images are in this batch
        batch_size = None
        for v in data.values():
            try:
                # tensors
                if hasattr(v, 'shape') and len(getattr(v, 'shape', ())) >= 1:
                    batch_size = int(v.shape[0])
                    break
                # lists/tuples
                if isinstance(v, (list, tuple)):
                    batch_size = len(v)
                    break
            except Exception:
                continue
        if batch_size is None:
            # fallback to samples_per_gpu
            batch_size = args.samples_per_gpu

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            model(return_loss=False, rescale=True, **data)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        # warmup counted in iterations
        if i >= num_warmup:
            pure_inf_time += elapsed
            images_processed += batch_size

            if images_processed >= total_images:
                # compute fps per image
                fps = images_processed / pure_inf_time if pure_inf_time > 0 else 0.0
                print(f'Overall fps: {fps:.2f} img / s')
                break

            if (images_processed) % args.log_interval == 0:
                fps = images_processed / pure_inf_time if pure_inf_time > 0 else 0.0
                print(f'Done images [{images_processed:<4}/ {total_images}], '
                      f'fps: {fps:.2f} img / s')


if __name__ == '__main__':
    main()
