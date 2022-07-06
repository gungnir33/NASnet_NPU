# Copyright (c) Soumith Chintala 2016,
# All rights reserved
#
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://spdx.org/licenses/BSD-3-Clause.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -*- coding: utf-8 -*-
"""用于精度比对
"""

import torch
import torchvision
from apex import amp
import copy

import pretrainedmodels
import pretrainedmodels.utils

##### 需自行改写部分 start #####
# 获得模型
def get_model():
    model = pretrainedmodels.nasnetamobile()
    return model

# 获得输入tensor
input_tensor = torch.randn(2, 3, 224, 224)

# 设置npu_device
npu_device = 'npu:0'

# 设置amp
AMP_MODE = True

# 设置NPU prof 文件输出
NPU_PROF = True

##### 需自行改写部分 start #####

# 设置hook
def hook_func(name, save_dict, module):
    def hook_function(module, inputs, outputs):
        inputs_key = name + '_inputs'
        idx = 0
        while inputs_key in save_dict:
            inputs_key = inputs_key.split('-')[0] + '-%d'%idx
            idx +=1
        save_dict[inputs_key] = inputs

        outputs_key = name + '_outputs'
        idx = 0
        while outputs_key in save_dict:
            outputs_key = outputs_key.split('-')[0] + '-%d'%idx
            idx +=1
        save_dict[outputs_key] = outputs
    return hook_function


##### CPU #####
# CPU固定输入和权重
model = get_model()
optimizer = torch.optim.SGD(model.parameters(), 0.1)
state_dict = copy.deepcopy(model.state_dict())

# CPU注册hook，cpu_dict用于存储对比对象
cpu_dict = {}
for name, module in model.named_modules():
    module.register_forward_hook(hook_func('[forward]:' + name, cpu_dict, module))
    module.register_backward_hook(hook_func('[backward]:' + name, cpu_dict, module))

# CPU运行正反向，获取正反向每个module的输入输出和所有参数的grad
out = model(input_tensor)
loss = out.mean()
optimizer.zero_grad()
loss.backward()
optimizer.step()
for name, param in model.named_parameters():
    cpu_dict["[grad]:" + name] = param.grad

##### NPU #####
# 重新定义模型，清理模型状态，并加装权重，保持初始化一致
model = get_model()
optimizer = torch.optim.SGD(model.parameters(), 0.1)
model.load_state_dict(state_dict)

# NPU注册hook，npu_dict用于存储对比对象
npu_dict = {}
for name, module in model.named_modules():
    module.register_forward_hook(hook_func('[forward]:' + name, npu_dict, module))
    module.register_backward_hook(hook_func('[backward]:' + name, npu_dict, module))

# 将model和input_tensor放到npu
torch.npu.set_device(npu_device)
model = model.npu()
input_tensor = input_tensor.npu()

# amp可选项，不适用请注释
if AMP_MODE:
    model, optimizer = amp.initialize(model, optimizer, opt_level='O2', loss_scale=1.0)

# NPU运行正反向，获取正反向每个module的输入输出和所有参数的grad
out = model(input_tensor)
loss = out.mean()
optimizer.zero_grad()
if AMP_MODE:
    with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()
else:
    loss.backward()
optimizer.step()
for name, param in model.named_parameters():
    npu_dict["[grad]:" + name] = param.grad


##### ComPare #####
# 递归得到对比值
def compare(x1, x2, prefix=''):
    if isinstance(x1, tuple):
        if x1:
            for idx in range(len(x1)):
                try:
                    compare(x1[idx], x2[idx], prefix=prefix + '.%d' % idx)
                except Exception as e:
                    print(str(e))
                    print(prefix, 'failed.')
    elif isinstance(x1, torch.Tensor) and isinstance(x2, torch.Tensor):
        try:
            l1_error = (x1 - x2.cpu()).abs().mean()
            rel_error = l1_error / (x1.abs().mean())
            print(prefix, 'l1_error: ', l1_error, 'rel_error', rel_error)
            if l1_error * rel_error > 10 :
                print('\n###\n',prefix, 'should checked!','\n###\n')
        except Exception as e:
            print(str(e))
            print(prefix, 'failed.')

for k in cpu_dict:
    compare(cpu_dict[k], npu_dict[k], prefix=k)

# 需要profiling的时候额外输出一次
if NPU_PROF:
    with torch.autograd.profiler.profile(use_npu=True) as prof:
        out = model(input_tensor)
        loss = out.mean()
        optimizer.zero_grad()
        if AMP_MODE:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
    prof.export_chrome_trace("output.prof")  # "output.prof"为输出文件地址

