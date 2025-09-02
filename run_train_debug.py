import sys, traceback
sys.path.insert(0, '.')


def main(use_cuda: bool = False, device: str | None = None):
    import importlib
    import mmengine
    from mmengine.runner import Runner
    from pathlib import Path

    # Resolve config path relative to this script so the script can be run from any CWD
    base_dir = Path(__file__).resolve().parent
    cfg_path = base_dir / 'local_configs' / 'segformer' / 'segformer_cityscapes_video.py'
    print('Loading config...')
    if not cfg_path.exists():
        # Fallback to plain relative path if the file isn't found where expected
        cfg_rel = Path('local_configs/segformer/segformer_cityscapes_video.py')
        if cfg_rel.exists():
            cfg_path = cfg_rel
        else:
            raise FileNotFoundError(f"Config file not found at '{cfg_path}' or '{cfg_rel}'")

    cfg = mmengine.Config.fromfile(str(cfg_path))

    if 'work_dir' not in cfg:
        cfg.work_dir = cfg.get('work_dir', 'work_dirs/segformer_cityscapes_video')

    # Remove incompatible train loop keys (some configs include max_iters which EpochBasedTrainLoop
    # does not accept). Keep max_epochs when present.
    try:
        if hasattr(cfg, 'train_cfg') and isinstance(cfg.train_cfg, dict):
            if 'max_iters' in cfg.train_cfg:
                print("Removing 'max_iters' from cfg.train_cfg (incompatible with EpochBasedTrainLoop)")
                cfg.train_cfg.pop('max_iters', None)
    except Exception:
        pass

    # Try to import mmseg dataset and transforms shim so worker processes can import them
    try:
        import mmseg.datasets.cityscapes_video as _city_ds
        print('✓ Imported mmseg.datasets.cityscapes_video')
    except Exception as e:
        print(f'⚠ Could not import mmseg.datasets.cityscapes_video: {e}')

    try:
        importlib.import_module('mmseg.datasets.transforms')
        print('✓ Imported mmseg.datasets.transforms')
    except Exception:
        try:
            importlib.import_module('tv3s_utils.utils.datasets.transforms')
            print('✓ Imported tv3s_utils utils transforms')
        except Exception as e:
            print(f'⚠ Could not import tv3s transforms: {e}')

    # Best-effort injection of shim transforms into registries used by Compose
    try:
        import mmcv.transforms.wrappers as _mmcv_wrappers
        import tv3s_utils.utils.datasets.transforms as _tv3s_shim

        _shim_map = {
            name: obj
            for name, obj in vars(_tv3s_shim).items()
            if isinstance(obj, type) and hasattr(obj, '__mro__') and _tv3s_shim._BaseTransform in obj.__mro__
        }

        moddict = getattr(_mmcv_wrappers.TRANSFORMS, '_module_dict', None)
        if isinstance(moddict, dict):
            for _n, _cls in _shim_map.items():
                if _n not in moddict:
                    moddict[_n] = _cls
            print('✓ Injected shim transforms into mmcv.transforms.wrappers.TRANSFORMS')
    except Exception:
        pass

    # Sanitise optimizer config (remove momentum for AdamW)
    try:
        def _sanitize_opt(opt_cfg):
            try:
                typ = opt_cfg.get('type') if hasattr(opt_cfg, 'get') else (opt_cfg.get('type') if isinstance(opt_cfg, dict) else None)
                if isinstance(typ, str) and typ.lower() == 'adamw':
                    if hasattr(opt_cfg, 'pop'):
                        opt_cfg.pop('momentum', None)
                    elif isinstance(opt_cfg, dict) and 'momentum' in opt_cfg:
                        del opt_cfg['momentum']
            except Exception:
                pass

        ow = cfg.get('optim_wrapper') if hasattr(cfg, 'get') else None
        if ow:
            opt = ow.get('optimizer') if hasattr(ow, 'get') else None
            if opt:
                _sanitize_opt(opt)
        top_opt = cfg.get('optimizer') if hasattr(cfg, 'get') else None
        if top_opt:
            _sanitize_opt(top_opt)
    except Exception:
        pass

    print('Creating Runner.from_cfg...')
    # For debug runs prefer main-process data loading to avoid worker import
    # complications and get clearer tracebacks. Force num_workers=0.
    try:
        if hasattr(cfg, 'train_dataloader') and isinstance(cfg.train_dataloader, dict):
            cfg.train_dataloader['num_workers'] = 0
            cfg.train_dataloader['persistent_workers'] = False
    except Exception:
        pass

    runner = Runner.from_cfg(cfg)
    print('Runner created')

    # One-time debug wrappers: inspect data_samples.metainfo when the model's
    # loss() and predict() are first called. This helps determine whether the
    # dataset instances expose `metainfo`/`dataset_meta` at runtime (the root
    # cause for metrics falling back to a single generated class).
    # One-time diagnostic: inspect the actual dataset instances used by the
    # runner's dataloaders (train/val/test) and print their class and any
    # metainfo-like attributes. This avoids wrapping model methods and is
    # safer across different mmengine/mmseg versions.
    try:
        def _inspect_loader(loader, name):
            try:
                ds = getattr(loader, 'dataset', None)
                print(f'DEBUG: {name} loader dataset type: {type(ds)}')
                if ds is None:
                    return
                # Common attribute names to check
                for attr in ('metainfo', 'dataset_meta', 'METAINFO', 'classes', 'CLASSES', 'PALETTE', 'palette'):
                    try:
                        val = getattr(ds, attr, None)
                        if val is not None:
                            print(f'DEBUG: {name} dataset.{attr} = {val}')
                    except Exception as _e:
                        print(f'DEBUG: error reading {attr} from {name} dataset: {_e}')
                # If dataset is a wrapper (e.g., Subset), try to reach inner dataset
                try:
                    inner = getattr(ds, 'dataset', None)
                    if inner is not None and inner is not ds:
                        print(f'DEBUG: {name} inner dataset type: {type(inner)}')
                        for attr in ('metainfo', 'dataset_meta', 'METAINFO', 'CLASSES'):
                            try:
                                val = getattr(inner, attr, None)
                                if val is not None:
                                    print(f'DEBUG: {name} inner.{attr} = {val}')
                            except Exception:
                                pass
                except Exception:
                    pass
            except Exception:
                pass

        try:
            # train loader
            _inspect_loader(getattr(runner, 'train_dataloader', None), 'train')
        except Exception:
            pass
        try:
            _inspect_loader(getattr(runner, 'val_dataloader', None), 'val')
        except Exception:
            pass
        try:
            _inspect_loader(getattr(runner, 'test_dataloader', None), 'test')
        except Exception:
            pass
    except Exception:
        pass

    print('Starting runner.train()...')
    try:
        # For debug runs we may want to avoid GPU OOMs. If CUDA is available,
        # move the model to CPU to ensure the first iterations complete and
        # produce deterministic, inspectable errors.
        try:
            import torch
            # If the user provided an explicit device, try to move the model there.
            if device is not None:
                try:
                    print(f'Moving model to device: {device}')
                    runner.model.to(device)
                except Exception:
                    pass
                try:
                    dp = getattr(runner.model, 'data_preprocessor', None)
                    if dp is not None and hasattr(dp, 'device'):
                        dp.device = device
                except Exception:
                    pass
            else:
                # Default debug behaviour: move model to CPU when CUDA is available
                # unless the user explicitly asked to use CUDA.
                if torch.cuda.is_available() and not use_cuda:
                    print('CUDA available: moving model to CPU for debug run to avoid OOM')
                    try:
                        runner.model.to('cpu')
                    except Exception:
                        pass
                    try:
                        # If the model has a data_preprocessor with a device field,
                        # update it so inputs are processed on CPU.
                        dp = getattr(runner.model, 'data_preprocessor', None)
                        if dp is not None and hasattr(dp, 'device'):
                            dp.device = 'cpu'
                    except Exception:
                        pass
                elif torch.cuda.is_available() and use_cuda:
                    print('CUDA available and --cuda requested: leaving model on CUDA')
                else:
                    # Either CUDA not available, or we're already on CPU — nothing to do.
                    pass
        except Exception:
            pass
        # On Windows multiprocessing spawn mode requires guarded main; we've used freeze_support below.
        runner.train()
    except Exception:
        print('Exception during runner.train():')
        traceback.print_exc()
        raise


if __name__ == '__main__':
    try:
        from multiprocessing import freeze_support
        freeze_support()
    except Exception:
        pass
    # Lightweight CLI so users can override debug behaviour
    try:
        import argparse

        parser = argparse.ArgumentParser(description='Run debug training with optional device control')
        parser.add_argument('--cuda', action='store_true', help='Allow using CUDA device if available (default: move to CPU for debug)')
        parser.add_argument('--device', type=str, default=None, help='Explicit device to move the model to, e.g. "cpu" or "cuda:0"')
        args, unknown = parser.parse_known_args()

        main(use_cuda=args.cuda, device=args.device)
    except Exception:
        # Fallback to previous behaviour
        main()
