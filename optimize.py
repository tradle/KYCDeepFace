# -*- coding: utf-8 -*-
# @Author: yirui
# @Date:   2021-09-01 17:39:28
# @Last Modified by:   yirui
# @Last Modified time: 2021-09-01 17:42:19
import onnx

# model = onnx.load("models/detection/fast_640.onnx")
# model = onnx.load("models/detection/hybrid_640.onnx")
# model = onnx.load("models/detection/landmarks_160.onnx")
model = onnx.load("models/detection/mask_64.onnx")
# model = onnx.load("models/recognition/mfn_112_96.onnx")

from onnxsim import simplify
# convert model
optimized_model, check = simplify(model)
assert check, "Simplified ONNX model could not be validated"
# onnx.save(optimized_model, "models/detection/fast_640_opt.onnx")
# onnx.save(optimized_model, "models/detection/hybrid_640_opt.onnx")
# onnx.save(optimized_model, "models/detection/landmarks_160_opt.onnx")
onnx.save(optimized_model, "models/detection/mask_64_opt.onnx")
# onnx.save(optimized_model, "models/recognition/mfn_112_96_opt.onnx")

# first trial
# passes = [ "eliminate_unused_initializer"]
# from onnx import optimizer
# optimized_model = optimizer.optimize(model, passes)

# onnx.save(optimized_model, "models/detection/fast_640_opt.onnx")


# if model.ir_version < 4:
#     print(
#         'Model with ir_version below 4 requires to include initilizer in graph input'
#     )


# second trial
# inputs = model.graph.input
# name_to_input = {}
# for input in inputs:
#     name_to_input[input.name] = input
# print(name_to_input)

# for initializer in model.graph.initializer:
#     print(initializer.name)
#     if initializer.name in name_to_input:
#         inputs.remove(name_to_input[initializer.name])

# print(inputs)

# onnx.save(model, "models/detection/fast_640_opt.onnx")
