// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file	loss_layer.cpp
 * @date	12 June 2020
 * @brief	This is Loss Layer Class for Neural Network
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <layer_context.h>
#include <loss_layer.h>

namespace nntrainer {
void LossLayer::finalize(InitLayerContext &context) {
  std::vector<TensorDim> input_dim = context.getInputDimensions();
  std::vector<TensorDim> output_dim = input_dim;

  for (auto &d : output_dim)
    d.setDataType(
      str_converter<enum_class_prop_tag,
                    nntrainer::TensorDataTypeInfo>::from_string("FP32"));

  context.setOutputDimensions(output_dim);

  is_inplace = true;
  if (context.getActivationDataType() != ml::train::TensorDim::DataType::FP32)
    is_inplace = false;
}

void LossLayer::updateLoss(RunLayerContext &context, const Tensor &l) {
  float loss_sum = 0.0f;
  const float *data = l.getData();

  for (unsigned int i = 0; i < l.batch(); i++) {
    loss_sum += data[i];
  }
  context.setLoss(loss_sum / (float)l.batch());
}

void LossLayer::applyLossScale(RunLayerContext &context, Tensor &ret_deriv) {

  float loss_scale = context.getLossScale();
  if (loss_scale != 1.0)
    ret_deriv.multiply_i(loss_scale);
}

/**
 * @copydoc Layer::setProperty(const std::vector<std::string> &values)
 */
void LossLayer::setProperty(const std::vector<std::string> &values) {
  NNTR_THROW_IF(!values.empty(), std::invalid_argument)
    << "[Layer] Unknown Layer Properties count = " << values.size();
}

} /* namespace nntrainer */
