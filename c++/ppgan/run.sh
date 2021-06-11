#!/bin/bash
set +x
set -e

work_path=$(dirname $(readlink -f $0))

# 1. compile
bash ${work_path}/compile.sh

# 2. run
./build/ppgan_test --inputs_file ./tmp/pix2pix_inputs.txt --model_file ./tmp/pix2pixmodel.pdmodel --params_file ./tmp/pix2pixmodel.pdiparams
mv output.txt pix2pix_output.txt
./build/ppgan_test --inputs_file ./tmp/cyclegan_inputs.txt --model_file ./tmp/cycleganmodel.pdmodel --params_file ./tmp/cycleganmodel.pdiparams
mv output.txt cyclegan_output.txt
./build/ppgan_test --inputs_file ./tmp/wav2lip_inputs.txt --model_file ./tmp/wav2lipmodelhq.pdmodel --params_file ./tmp/wav2lipmodelhq.pdiparams
mv output.txt wav2lip_output.txt
./build/ppgan_test --inputs_file ./tmp/edvr_inputs.txt --model_file ./tmp/edvrmodel.pdmodel --params_file ./tmp/edvrmodel.pdiparams
mv output.txt edvr_output.txt
./build/ppgan_test --inputs_file ./tmp/esrgan_inputs.txt --model_file ./tmp/esrgan.pdmodel --params_file ./tmp/esrgan.pdiparams
mv output.txt esrgan_output.txt

# 3. convert to image
python txt2img.py  --model_type pix2pix --input_file pix2pix_output.txt
python txt2img.py  --model_type cyclegan --input_file cyclegan_output.txt
python txt2img.py  --model_type wav2lip --input_file wav2lip_output.txt
python txt2img.py  --model_type edvr --input_file edvr_output.txt
python txt2img.py  --model_type esrgan --input_file esrgan_output.txt