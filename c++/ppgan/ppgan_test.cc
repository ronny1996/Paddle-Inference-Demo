#include <chrono>
#include <memory>
#include <numeric>
#include <cmath>
#include <sstream>
#include <iostream>
#include <fstream>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "paddle/include/paddle_inference_api.h"

using paddle_infer::Config;
using paddle_infer::Predictor;
using paddle_infer::CreatePredictor;

DEFINE_string(model_file, "", "Directory of the inference model.");
DEFINE_string(params_file, "", "Directory of the inference model.");
DEFINE_string(model_dir, "", "Directory of the inference model.");
DEFINE_int32(warmup, 0, "warmup.");
DEFINE_int32(repeats, 1, "repeats.");
DEFINE_bool(use_gpu, false, "use gpu.");
DEFINE_string(inputs_file, "", "Path of the inputs file.");

using Time = decltype(std::chrono::high_resolution_clock::now());
Time time() { return std::chrono::high_resolution_clock::now(); };
double time_diff(Time t1, Time t2) {
  typedef std::chrono::microseconds ms;
  auto diff = t2 - t1;
  ms counter = std::chrono::duration_cast<ms>(diff);
  return counter.count() / 1000.0;
}

std::shared_ptr<Predictor> InitPredictor() {
  Config config;
  if (FLAGS_model_dir != "") {
    config.SetModel(FLAGS_model_dir);
  }
  config.SetModel(FLAGS_model_file, FLAGS_params_file);
  if (FLAGS_use_gpu) {
    config.EnableUseGpu(100, 0);
  } else {
    config.EnableMKLDNN();
  }

  // Open the memory optim.
  config.EnableMemoryOptim();
  return CreatePredictor(config);
}

void FeedFromFile(std::shared_ptr<Predictor> predictor, std::string& inputs_file) {
  auto input_names = predictor->GetInputNames();
  std::vector<std::vector<float>> inputs;
  std::fstream in_file(inputs_file, std::ios_base::in);
  std::vector<std::vector<int>> inputs_size;

  if (!in_file.is_open()) {
    std::cout << "Failed to open " << inputs_file << std::endl;
    exit(-1);
  }
  while (!in_file.eof()) {
    std::string line;

    std::getline(in_file, line);
    line = line.erase(line.find_last_not_of(" ") + 1);
    line = line.erase(0, line.find_first_not_of(" "));
    if (line.empty()) break;
    std::vector<int> input_size;
    std::stringstream ss1(line);
    while (ss1) {
      int data;
      ss1 >> data;
      if (ss1.fail()) break;
      input_size.emplace_back(data);
    }
    inputs_size.emplace_back(input_size);

    std::getline(in_file, line);
    line = line.erase(line.find_last_not_of(" ") + 1);
    line = line.erase(0, line.find_first_not_of(" "));
    if (line.empty()) break;
    std::vector<float> input;
    std::stringstream ss2(line);
    while (ss2) {
      float data;
      ss2 >> data;
      if (ss2.fail()) break;
      input.emplace_back(data);
    }
    inputs.emplace_back(input);
  }
  if (inputs_size.size() != inputs.size()) {
    std::cout << "inputs_size.size != inputs.size\n";
    exit(-1);
  }
  for (auto i = 0; i < inputs_size.size(); i++) {
    auto numel = std::accumulate(inputs_size[i].begin(), inputs_size[i].end(), 1, std::multiplies<int>());
    std::cout << "Inputs[" << i << "] size: " << inputs[i].size() << std::endl;
    if (numel != inputs[i].size()) {
      std::cout << "inputs[" << i << "].size() != " << numel << "\n";
      exit(-1);
    }
  }

  for (size_t i = 0; i < input_names.size(); ++i) {
    auto input_t = predictor->GetInputHandle(input_names[i]);
    input_t->Reshape(inputs_size[i]);
    input_t->CopyFromCpu(inputs[i].data());
  }
  for (size_t i = 0; i < FLAGS_warmup; ++i)
    CHECK(predictor->Run());
}

void run(std::shared_ptr<Predictor> predictor, std::vector<float>& output_data, std::vector<int>& output_shape) {
  auto output_names = predictor->GetOutputNames();
  auto st = time();
  for (size_t i = 0; i < FLAGS_repeats; ++i) {
    CHECK(predictor->Run());
    auto output_t = predictor->GetOutputHandle(output_names[0]);
    output_shape = output_t->shape();
    int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                  std::multiplies<int>());
    output_data.resize(out_num);
    output_t->CopyToCpu(output_data.data());
  }
  LOG(INFO) << "run avg time is " << time_diff(st, time()) / FLAGS_repeats
            << " ms";
}

int main(int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);

  auto predictor = InitPredictor();
  FeedFromFile(predictor, FLAGS_inputs_file);

  std::vector<int> output_shape;
  std::vector<float> output_data;
  run(predictor, output_data, output_shape);

  std::fstream out_file("./output.txt", std::ios_base::out);
  for (auto& t : output_shape) {
    out_file << t;
    out_file << " ";
  }
  out_file << std::endl;
  for (auto& t : output_data) {
    out_file << t;
    out_file << " ";
  }
  return 0;
}