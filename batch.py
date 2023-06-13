# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import multiprocessing as mp
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import utils
from demo.predictor import VisualizationDemo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from moviepy.editor import ImageSequenceClip
from tqdm import tqdm


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()

    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = (
        args.confidence_threshold
    )
    cfg.freeze()

    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")

    parser.add_argument("input", type=str, help="The video input directory.", nargs="?")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )

    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )

        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()

        if os.path.isfile(filename):
            return True

        return False


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    args = get_parser().parse_args()
    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg, parallel=True, person_only=True)

    config_file = args.config_file.split("/")[-1].split(".")[0]
    input_path = Path(args.input)
    output_path = input_path.parent / f"{input_path.name}-{config_file}"
    n_files = utils.count_files(input_path)
    count = 0
    video_limit = 1

    for file in input_path.rglob(f"*.mp4"):
        count += 1

        print(f"{count}/{video_limit if video_limit != -1 else n_files}")

        video = cv2.VideoCapture(str(file))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        relative_path = file.relative_to(str(input_path))
        output_video_path = output_path / relative_path.with_suffix(".mp4")

        output_video_path.parent.mkdir(parents=True, exist_ok=True)

        output_frames = []
        viz = demo.run_on_video(video)

        for output_frame, instances in tqdm(viz, total=num_frames):
            output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)

            output_frames.append(output_frame)

        ImageSequenceClip(output_frames, fps=fps).write_videofile(
            str(output_video_path), logger=None, audio=False
        )

        if count == video_limit:
            break
