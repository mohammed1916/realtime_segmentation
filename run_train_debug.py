import sys, traceback
sys.path.insert(0, '.')


def main():
    import importlib
    import mmengine
    from mmengine.runner import Runner

    print('Loading config...')
    cfg = mmengine.Config.fromfile('local_configs/segformer/segformer_cityscapes_video.py')

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

    print('Starting runner.train()...')
    try:
        # For debug runs we may want to avoid GPU OOMs. If CUDA is available,
        # move the model to CPU to ensure the first iterations complete and
        # produce deterministic, inspectable errors.
        try:
            import torch
            if torch.cuda.is_available():
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
    main()
