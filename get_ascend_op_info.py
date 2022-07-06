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
"""用于导出OPINFO
"""

import os
import sys
import argparse


def func(host_log_folder):
    """
    :param host_log_folder: where host_log_folder addr is.
    :return:
    """
    host_log_files = os.listdir(host_log_folder)
    result = []

    for host_log in host_log_files:
        if not (host_log.endswith('.log') or host_log.endswith('.out')):
            continue
        with open(os.path.join(host_log_folder, host_log), 'r', encoding='utf-8')as f:
            host_log_lines = f.readlines()
            for line in host_log_lines:
                if line.startswith('[INFO] ASCENDCL') and "aclopCompile::aclOp" in line:
                    op_info = line.split('OpType: ')[1][:-2]
                    op_type = op_info.split(',')[0]
                    op_param = op_info[len(op_type) + 2:]

                    result.append(op_type + ' ' + op_param)

    with open('ascend_op_info_summary.txt', 'w')as f:
        for i in result:
            f.write(i+'\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='trans the log')
    parser.add_argument('--host_log_folder', default="./",
                        help="input the dir name, trans the current dir with default")
    ags = parser.parse_args()
    func(ags.host_log_folder)