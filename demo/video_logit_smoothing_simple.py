# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import cv2

from mmengine.model import revert_sync_batchnorm
from mmseg.apis import inference_model, init_model, show_result_pyplot


def main():
    parser = ArgumentParser()
    parser.add_argument('video', help='Video file or webcam id')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--show', action='store_true', help='Whether to show results live')
    parser.add_argument('--show-wait-time', default=1, type=int, help='Wait time for cv2.imshow')
    parser.add_argument('--output-file', default=None, type=str, help='Output video file path')
    parser.add_argument('--output-fourcc', default='MJPG', type=str, help='Fourcc of output video')
    parser.add_argument('--output-fps', default=-1, type=int, help='FPS of output video')
    parser.add_argument('--output-height', default=-1, type=int, help='Frame height')
    parser.add_argument('--output-width', default=-1, type=int, help='Frame width')
    parser.add_argument('--opacity', type=float, default=0.5, help='Opacity of segmentation overlay')
    args = parser.parse_args()

    assert args.show or args.output_file, 'At least one output should be enabled.'

    # build model
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)

    # open video
    if args.video.isdigit():
        args.video = int(args.video)
    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened(), f'Failed to open {args.video}'

    input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_fps = cap.get(cv2.CAP_PROP_FPS)

    # setup output writer
    writer = None
    if args.output_file is not None:
        fourcc = cv2.VideoWriter_fourcc(*args.output_fourcc)
        output_fps = args.output_fps if args.output_fps > 0 else (input_fps if input_fps > 0 else 25)
        output_height = args.output_height if args.output_height > 0 else input_height
        output_width = args.output_width if args.output_width > 0 else input_width
        writer = cv2.VideoWriter(args.output_file, fourcc, output_fps,
                                 (output_width, output_height), True)

    try:
        while True:
            flag, frame = cap.read()
            if not flag:
                break

            # MMSeg handles preprocessing internally
            with torch.no_grad():
                result = inference_model(model, frame)

            # visualize
            draw_img = show_result_pyplot(
                model,
                frame,
                result,
                opacity=args.opacity,
                show=False
            )

            if args.show:
                cv2.imshow('video_demo', draw_img)
                if cv2.waitKey(args.show_wait_time) & 0xFF == ord('q'):
                    break
            if writer:
                if draw_img.shape[0] != input_height or draw_img.shape[1] != input_width:
                    draw_img = cv2.resize(draw_img, (input_width, input_height))
                writer.write(draw_img)
    finally:
        if writer:
            writer.release()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    import torch  # keep torch import inside to match your script style
    main()
