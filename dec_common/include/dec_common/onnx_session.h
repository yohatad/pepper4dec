/* onnx_session.h
 *
 * Create an ONNX Runtime session on the GPU when possible, on the CPU
 * otherwise. Header-only, so dec_common itself does not link ONNX Runtime;
 * include it after <onnxruntime_cxx_api.h>.
 *
 * Appending the CUDA execution provider usually succeeds even where CUDA is
 * unusable (no device visible, a CPU-only ONNX Runtime build that still
 * exports the call, mismatched CUDA libraries); the failure then surfaces
 * when the session is created. So the fallback has to wrap session creation,
 * not just AppendExecutionProvider_CUDA, or the node fails to configure on a
 * machine without a working GPU.
 *
 * Author: Yohannes Tadesse Haile
 * Affiliation: Carnegie Mellon University Africa
 * Date: November 13, 2026
 * Version: v1.0
 *
 * Copyright (C) 2025 Carnegie Mellon University Africa
 * This software is provided 'as-is' for research and educational purposes
 * within the DEC project.
 */

#pragma once

#include <memory>
#include <string>
#include <thread>

namespace dec_common
{

/// Session options shared by every DEC model: all cores, full graph
/// optimisation, and no busy-waiting between runs.
inline Ort::SessionOptions baseSessionOptions()
{
  Ort::SessionOptions options;
  options.SetIntraOpNumThreads(static_cast<int>(std::thread::hardware_concurrency()));
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  // ONNX Runtime's worker threads spin by default while waiting for work. With
  // one thread per core that kept every core busy (2700% CPU on a 28-core
  // laptop for 15 frames/s of 28 ms inference), starving the other nodes and
  // the camera feed; person_detection then lost a third of its frames.
  options.AddConfigEntry("session.intra_op.allow_spinning", "0");
  options.AddConfigEntry("session.inter_op.allow_spinning", "0");
  return options;
}

/**
 * Create a session for @p model_path, trying CUDA first when @p try_cuda.
 * If CUDA cannot be appended or the CUDA session cannot be created, the
 * session is rebuilt on the CPU. @p used_cuda reports which one was made;
 * @p cuda_error receives the reason CUDA was given up, if it was.
 * Throws only if the CPU session cannot be created either (e.g. a bad model).
 */
inline std::unique_ptr<Ort::Session> createSession(
  Ort::Env & env, const std::string & model_path, bool try_cuda,
  bool & used_cuda, std::string & cuda_error)
{
  used_cuda = false;
  cuda_error.clear();
  if (try_cuda) {
    try {
      Ort::SessionOptions options = baseSessionOptions();
      OrtCUDAProviderOptions cuda_options{};
      options.AppendExecutionProvider_CUDA(cuda_options);
      auto session = std::make_unique<Ort::Session>(env, model_path.c_str(), options);
      used_cuda = true;
      return session;
    } catch (const std::exception & e) {
      cuda_error = e.what();
    }
  }
  Ort::SessionOptions options = baseSessionOptions();
  return std::make_unique<Ort::Session>(env, model_path.c_str(), options);
}

}  // namespace dec_common
