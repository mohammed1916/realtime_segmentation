# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import cv2
import torch
import numpy as np
from mmengine.model.utils import revert_sync_batchnorm
from mmseg.apis import init_model
from mmseg.visualization import SegLocalVisualizer
from mmseg.structures import SegDataSample
from mmengine.structures import PixelData
import re

def get_stream_url(url):
    """Get stream URL from YouTube or return the URL if it's already a stream."""
    try:
        import streamlink
        streams = streamlink.streams(url)
        if streams:
            stream = streams.get('best') or streams.get('720p') or streams.get('480p') or list(streams.values())[0]
            return stream.url
        else:
            return url
    except ImportError:
        print("streamlink not installed. Please install with: pip install streamlink")
        return url
    except Exception as e:
        print(f"Error getting stream URL: {e}")
        return url

def main():
    parser = ArgumentParser()
    parser.add_argument('video', help='Video file, webcam id, YouTube URL, or stream URL')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--show', action='store_true', help='Whether to show result')
    parser.add_argument('--show-wait-time', default=1, type=int, help='Wait time after imshow')
    parser.add_argument('--output-file', default=None, type=str, help='Output video file path')
    parser.add_argument('--output-fourcc', default='MJPG', type=str, help='Fourcc of output video')
    parser.add_argument('--output-fps', default=-1, type=int, help='FPS of output video')
    parser.add_argument('--output-height', default=-1, type=int, help='Frame height')
    parser.add_argument('--output-width', default=-1, type=int, help='Frame width')
    parser.add_argument('--opacity', type=float, default=0.5, help='Opacity of painted map')
    parser.add_argument('--smooth-alpha', type=float, default=0.6,
                        help='Weight for previous logits in smoothing (0-1)')
    args = parser.parse_args()

    assert args.show or args.output_file, 'At least one output should be enabled.'

    # build the model
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)

    # Image normalization config (from ADE20K)
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True
    )

    # Initialize visualizer
    visualizer = SegLocalVisualizer(
        vis_backends=[dict(type='LocalVisBackend')],
        save_dir=None,
        alpha=args.opacity
    )

    # Determine video source
    video_source = args.video
    if args.video.isdigit():
        video_source = int(args.video)
    elif re.match(r'https?://', args.video):
        # It's a URL
        if 'youtube.com' in args.video or 'youtu.be' in args.video:
            print("Detected YouTube URL, getting stream URL...")
            video_source = get_stream_url(args.video)
        else:
            # Direct stream URL
            video_source = args.video

    # open video
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Failed to open video source: {video_source}")
        return
    input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_fps = cap.get(cv2.CAP_PROP_FPS)

    # output video init
    writer = None
    output_height = input_height
    output_width = input_width
    if args.output_file is not None:
        fourcc = cv2.VideoWriter_fourcc(*args.output_fourcc)
        output_fps = args.output_fps if args.output_fps > 0 else input_fps
        output_height = args.output_height if args.output_height > 0 else input_height
        output_width = args.output_width if args.output_width > 0 else input_width
        writer = cv2.VideoWriter(args.output_file, fourcc, output_fps,
                                 (output_width, output_height), True)

    prev_logits = None  # store previous frame logits
    frame_count = 0

    try:
        while True:
            flag, frame = cap.read()
            if not flag:
                break

            frame_count += 1
            print(f"Processing frame {frame_count}...")

            # preprocess image manually
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            img = (img - np.array(img_norm_cfg['mean']) / 255.0) / (np.array(img_norm_cfg['std']) / 255.0)
            img = img.transpose(2, 0, 1)  # HWC to CHW
            input_tensor = torch.from_numpy(img).float().unsqueeze(0).to(args.device)

            # prepare metadata
            batch_img_metas = [{
                'ori_shape': frame.shape[:2],  # (H, W)
                'img_shape': img.shape[1:],    # (H, W) after preprocessing
                'pad_shape': img.shape[1:],    # (H, W)
                'scale_factor': 1.0,
                'flip': False
            }]

            # run inference
            with torch.no_grad():
                result = model.inference(input_tensor, batch_img_metas=batch_img_metas)

            # get logits tensor
            logits = result[0].float()  # [num_classes, H, W]

            # apply temporal smoothing
            if prev_logits is not None:
                logits = (1 - args.smooth_alpha) * logits + args.smooth_alpha * prev_logits
            prev_logits = logits.clone()

            # create result for visualization
            pred_seg = logits.argmax(dim=0).cpu().numpy()
            result_sample = SegDataSample()
            result_sample.pred_sem_seg = PixelData(data=pred_seg)

            # visualize
            visualizer.add_datasample(
                'video_demo',
                frame,
                result_sample,
                show=False,
                out_file=None,
                draw_gt=False,
                draw_pred=True
            )

            # get the visualized image
            draw_img = visualizer.get_image()

            if args.show:
                cv2.imshow('video_demo', draw_img)
                if cv2.waitKey(args.show_wait_time) & 0xFF == ord('q'):
                    break
            if writer:
                if draw_img.shape[0] != output_height or draw_img.shape[1] != output_width:
                    draw_img = cv2.resize(draw_img, (output_width, output_height))
                writer.write(draw_img)

    finally:
        if writer:
            writer.release()
        cap.release()
        cv2.destroyAllWindows()
        print(f"Processed {frame_count} frames successfully!")


if __name__ == '__main__':
    main()
