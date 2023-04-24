import numpy as np
import argparse
import cv2

from paddle.inference import Config
from paddle.inference import create_predictor
from paddle.inference import PrecisionType

from img_preprocess import preprocess


def init_predictor(args):
    if args.model_dir is not "":
        config = Config(args.model_dir)
    else:
        config = Config(args.model_file, args.params_file)

    config.enable_memory_optim()

    gpu_precision = PrecisionType.Float32
    if args.run_mode == "gpu_fp16":
        gpu_precision = PrecisionType.Half

    config.enable_use_gpu(1000, 0, gpu_precision)

    if args.run_mode == "trt_fp32":
        config.enable_tensorrt_engine(
            workspace_size=1 << 30,
            max_batch_size=1,
            min_subgraph_size=5,
            precision_mode=PrecisionType.Float32,
            use_static=False,
            use_calib_mode=False,
        )
    elif args.run_mode == "trt_fp16":
        config.enable_tensorrt_engine(
            workspace_size=1 << 30,
            max_batch_size=1,
            min_subgraph_size=5,
            precision_mode=PrecisionType.Half,
            use_static=False,
            use_calib_mode=False,
        )
    elif args.run_mode == "trt_int8":
        config.enable_tensorrt_engine(
            workspace_size=1 << 30,
            max_batch_size=1,
            min_subgraph_size=5,
            precision_mode=PrecisionType.Int8,
            use_static=False,
            use_calib_mode=True,
        )
    if args.use_dynamic_shape:
        names = ["inputs"]
        min_input_shape = [[1, 3, 112, 112]]
        max_input_shape = [[1, 3, 448, 448]]
        opt_input_shape = [[1, 3, 224, 224]]

        config.set_trt_dynamic_shape_info(
            {names[0]: min_input_shape[0]},
            {names[0]: max_input_shape[0]},
            {names[0]: opt_input_shape[0]},
        )

    predictor = create_predictor(config)
    return predictor


def run(predictor, img):
    # copy img data to input tensor
    input_names = predictor.get_input_names()
    for i, name in enumerate(input_names):
        input_tensor = predictor.get_input_handle(name)
        input_tensor.reshape(img[i].shape)
        input_tensor.copy_from_cpu(img[i])

    # do the inference
    predictor.run()

    results = []
    # get out data from output tensor
    output_names = predictor.get_output_names()
    for i, name in enumerate(output_names):
        output_tensor = predictor.get_output_handle(name)
        output_data = output_tensor.copy_to_cpu()
        results.append(output_data)

    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_file",
        type=str,
        default="",
        help="Model filename, Specify this when your model is a combined model.",
    )
    parser.add_argument(
        "--params_file",
        type=str,
        default="",
        help="Parameter filename, Specify this when your model is a combined model.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="",
        help="Model dir, If you load a non-combined model, specify the directory of the model.",
    )
    parser.add_argument(
        "--run_mode",
        type=str,
        default="",
        help="Run_mode which can be: trt_fp32, trt_fp16, trt_int8 and gpu_fp16.",
    )
    parser.add_argument(
        "--use_dynamic_shape",
        type=int,
        default=0,
        help="Whether use trt dynamic shape.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pred = init_predictor(args)
    img = cv2.imread("./ILSVRC2012_val_00000247.jpeg")
    img = preprocess(img)
    # img = np.ones((1, 3, 224, 224)).astype(np.float32)
    result = run(pred, [img])
    print("class index: ", np.argmax(result[0][0]))
