import os
import sys
import traceback
import mmseg 

sys.path.insert(0, '.')

def _get_logger():
    from mmengine.logging import MMLogger
    logger = MMLogger.get_current_instance()
    return logger


def _get_last_checkpoint_path(cfg):
    """Return the absolute path to the checkpoint referenced by cfg.work_dir/last_checkpoint(.pth).

    If the file contains a relative filename (common), join it with cfg.work_dir.
    Returns None when not found or unreadable.
    """
    work_dir = None
    if hasattr(cfg, 'work_dir'):
        work_dir = cfg.work_dir
    else:
        try:
            work_dir = cfg.get('work_dir', None)
        except Exception:
            work_dir = None

    if not work_dir:
        return None

    base = os.path.join(str(work_dir), 'last_checkpoint')
    if os.path.exists(base):
        p = base
    elif os.path.exists(base + '.pth'):
        p = base + '.pth'
    else:
        return None

    try:
        with open(p, 'r') as f:
            content = f.read().strip()
        if not content:
            return None
        # If content is a bare filename, join with work_dir
        if os.path.isabs(content):
            return content
        return os.path.join(str(work_dir), content)
    except Exception:
        return None


def _verify_loaded_weights(runner, ckpt_state_dict):
    """Quickly compare one overlapping parameter between checkpoint and model.

    Prints a simple PASS/WARN message.
    """
    try:
        import torch as _torch

        ms = runner.model.state_dict()
        common = [k for k in ckpt_state_dict.keys() if k in ms]
        if not common:
            print('VERIFY: no overlapping keys to compare')
            return
        k = common[0]
        ck_t = ckpt_state_dict[k].cpu()
        md_t = ms[k].cpu()
        # shapes may differ if optimizer-state-only etc.; guard the comparison
        if ck_t.shape == md_t.shape and _torch.allclose(ck_t, md_t, atol=1e-6):
            print(f'VERIFY PASS: parameter {k} matches checkpoint')
        else:
            print(f'VERIFY WARN: parameter {k} differs between checkpoint and model')
    except Exception as e:
        print('VERIFY: comparison failed:', e)


def _load_checkpoint_only_last(runner, cfg):
    """Attempt to load only the checkpoint referenced by cfg.work_dir/last_checkpoint(.pth).

    Preferred order:
      1. runner.load_checkpoint(path)
      2. torch.load(path, weights_only=True) -> apply state_dict to model
      3. Controlled allowlist of mmengine.logging.history_buffer.HistoryBuffer then torch.load(weights_only=False)

    Returns True if any weights were applied to the model, False otherwise.
    """
    logger = _get_logger()
    def L(msg):
        if logger:
            try:
                logger.info(msg)
            except Exception:
                print(msg)
        else:
            print(msg)

    path = _get_last_checkpoint_path(cfg)
    if not path:
        L('No last_checkpoint file found in cfg.work_dir; skipping checkpoint load.')
        return False

    L(f'Loading checkpoint pointed by last_checkpoint: {path}')

    try:
        # 1) Prefer runner.load_checkpoint (it may handle mmengine-specific objects)
        try:
            ck = runner.load_checkpoint(path)
            L(f'runner.load_checkpoint succeeded: {path}')
            st = ck.get('state_dict', ck)
            if isinstance(st, dict):
                _verify_loaded_weights(runner, st)
            return True
        except Exception as e1:
            L(f'runner.load_checkpoint failed: {e1}')

        # 2) Try torch.load weights_only
        import torch
        try:
            ck2 = torch.load(path, map_location='cpu', weights_only=True)
            st2 = ck2.get('state_dict', ck2)
            if isinstance(st2, dict):
                runner.model.load_state_dict(st2, strict=False)
                L('Weights-only torch.load applied to model')
                _verify_loaded_weights(runner, st2)
                return True
        except Exception as e2:
            L(f'weights-only torch.load failed: {e2}')
            # Detect HistoryBuffer issue and attempt allowlist only when it appears
            if 'HistoryBuffer' in str(e2) or 'mmengine.logging.history_buffer' in str(e2):
                try:
                    import mmengine.logging.history_buffer as _hb
                    hb = getattr(_hb, 'HistoryBuffer', None)
                    if hb is not None:
                        try:
                            # Prefer the context manager safe_globals when available
                            if hasattr(torch.serialization, 'safe_globals'):
                                with torch.serialization.safe_globals([hb]):
                                    ck3 = torch.load(path, map_location='cpu', weights_only=False)
                            else:
                                # older PyTorch exposes add_safe_globals which registers globals
                                torch.serialization.add_safe_globals([hb])
                                ck3 = torch.load(path, map_location='cpu', weights_only=False)
                            st3 = ck3.get('state_dict', ck3)
                            if isinstance(st3, dict):
                                runner.model.load_state_dict(st3, strict=False)
                                L('Loaded checkpoint after allowlisting HistoryBuffer')
                                _verify_loaded_weights(runner, st3)
                                return True
                        except Exception as e3:
                            L(f'Allowlist load failed: {e3}')
                    else:
                        L('HistoryBuffer not available for allowlisting')
                except Exception as e_imp:
                    L(f'Failed to import HistoryBuffer for allowlisting: {e_imp}')

        L('No recoverable load; continuing with freshly initialized model')
        return False

    except Exception as e:
        L(f'Unexpected error during checkpoint loading: {e}')
        if os.environ.get('DEBUG_DUMP_EXC'):
            traceback.print_exc()
        return False


def main(use_cuda=False, device=None):
    # lightweight environment setup
    try:
        import mmseg
        from mmseg.utils import register_all_modules
    except Exception:
        # mmseg may already be available through PYTHONPATH
        register_all_modules = None

    try:
        import mmengine
        from mmengine.runner import Runner
    except Exception as e:
        print('mmengine or Runner import failed:', e)
        raise

    # Register mmseg modules when available
    try:
        if register_all_modules:
            register_all_modules()
    except Exception:
        pass
    # Load the config used for training
    cfg_path = 'local_configs/segformer/segformer_cityscapes_video.py'
    if not os.path.exists(cfg_path):
        print(f'Config file not found: {cfg_path}')
        return
    cfg = mmengine.Config.fromfile(cfg_path)
    print("Model num_classes:", cfg.model.decode_head.num_classes)


    # Ensure work_dir exists
    work_dir = getattr(cfg, 'work_dir', None)
    if not work_dir:
        work_dir = 'work_dirs/segformer_cityscapes_video'
        cfg.work_dir = work_dir
    os.makedirs(str(work_dir), exist_ok=True)

    # Build the runner
    runner = Runner.from_cfg(cfg)

    # Attempt to load only the last checkpoint file
    _load_checkpoint_only_last(runner, cfg)

    # If user only wanted to test loading, exit early
    if os.environ.get('DEBUG_LOAD_ONLY'):
        print('DEBUG_LOAD_ONLY set: exiting after checkpoint load/verify (no training).')
        return

    # Move model based on device flags
    try:
        import torch
        if device:
            try:
                runner.model.to(device)
            except Exception:
                pass
        else:
            if torch.cuda.is_available() and not use_cuda:
                print('CUDA available: moving model to CPU for debug run to avoid OOM')
                try:
                    runner.model.to('cpu')
                except Exception:
                    pass
            elif torch.cuda.is_available() and use_cuda:
                print('CUDA available and --cuda requested: leaving model on CUDA')
    except Exception:
        pass

    print('Starting runner.train()...')
    try:
        runner.train()

    except KeyboardInterrupt:
        print('🚩 Training interrupted by user')
    except Exception:
        print('Exception during runner.train():')
        traceback.print_exc()
        raise


if __name__ == '__main__':
    try:
        from multiprocessing import freeze_support
        freeze_support()


        import argparse

        parser = argparse.ArgumentParser(description='Run debug training with optional device control')
        parser.add_argument('--cuda', action='store_true', help='Allow using CUDA device if available (default: move to CPU for debug)')
        parser.add_argument('--device', type=str, default=None, help='Explicit device to move the model to, e.g. "cpu" or "cuda:0"')
        args, unknown = parser.parse_known_args()

        main(use_cuda=args.cuda, device=args.device)
    except KeyboardInterrupt:
        print('🚩🚩🚩🚩🚩🚩🚩 Training interrupted by user')
        import sys
        sys.exit(0)
    