// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   preprocess_l2norm_layer.cpp
 * @date   09 Jan 2021
 * @brief  This file contains the simple l2norm layer which normalizes
 * the given feature
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <iostream>
#include <regex>
#include <sstream>

#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <preprocess_l2norm_layer.h>

namespace nntrainer {
static constexpr size_t SINGLE_INOUT_IDX = 0;

void PreprocessL2NormLayer::finalize(InitLayerContext &context) {
  const auto &input_dim = context.getInputDimensions()[0];
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "l2norm layer is designed for a single input only";
  NNTR_THROW_IF(input_dim.channel() != 1 || input_dim.height() != 1,
                std::invalid_argument)
    << "l2norm layer is designed for channel and height is 1 for now, please "
       "check";

  context.setOutputDimensions(context.getInputDimensions());
}

void PreprocessL2NormLayer::forwarding(RunLayerContext &context,
                                       bool training) {
  auto &hidden_ = context.getOutput(SINGLE_INOUT_IDX);

  auto &input_ = context.getInput(SINGLE_INOUT_IDX);

  for (unsigned int b = 0; b < input_.batch(); ++b) {
    auto input_slice = input_.getBatchSlice(b, 1);
    auto hidden_slice = hidden_.getBatchSlice(b, 1);
    input_slice.multiply(1 / input_slice.l2norm(), hidden_slice);
  }
}

void PreprocessL2NormLayer::calcDerivative(RunLayerContext &context) {
  throw std::invalid_argument("[L2Norm::calcDerivative] This Layer "
                              "does not support backward propagation");
}

void PreprocessL2NormLayer::setProperty(
  const std::vector<std::string> &values) {
  if (!values.empty()) {
    std::string msg = "[FlattenLayer] Unknown Layer Properties count " +
                      std::to_string(values.size());
    throw exception::not_supported(msg);
  }
}

} // namespace nntrainer
