import paddle
import argparse
import numpy as np

from ppgan.utils.config import get_config
from ppgan.datasets.builder import build_dataloader
from ppgan.engine.trainer import IterLoader
from ppgan.utils.visual import save_image
from ppgan.utils.visual import tensor2img
from ppgan.utils.filesystem import makedirs

MODEL_CLASSES = ["pix2pix", "cyclegan", "wav2lip", "esrgan", "edvr"]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        default=None,
        type=str,
        required=True)
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES))
    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.input_file) as f:
        line = f.readline()
        data_shape = np.fromstring(line, dtype=int, sep=' ')
        line = f.readline()
        data = np.fromstring(line, dtype=float, sep=' ')
        data = data.reshape(data_shape)

    if args.model_type in ["wav2lip", "esrgan", "edvr"]:
        min_max = (0, 1)
    else:
        min_max = (-1, 1)

    if args.model_type == "wav2lip":
        for j in range(data.shape[0]):
            data[j] = data[j][::-1, :, :]
            image_numpy = tensor2img(paddle.to_tensor(data[j]), min_max)
            save_image(image_numpy, args.model_type + "_output_{}.png".format(j))
    else:
        data = paddle.to_tensor(data[0])
        image_numpy = tensor2img(data, min_max)
        save_image(image_numpy, args.model_type + "_output.png")

if __name__ == '__main__':
    main()
