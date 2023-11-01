# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from distutils.log import error
from ntpath import join
import pandas as pd
import matplotlib.pyplot as plt
import os 
import math
import numpy as np
from scipy.interpolate import interp1d
from ast import literal_eval

# Text and environment settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams.update({'font.size': 15})
plt.figure(figsize=(20, 10))



cot_vel_max_gym = 4.0
cot_vel_step = 0.1



df_folder = '20231101-213543_vel'
df_file = 'df_vel'


root_dir = '/home/dklee98/git/ai707_ws/ViNL/logs/'
df_all = pd.read_csv(root_dir + df_folder+'/'+df_file+'.csv', index_col=0)
plt_x = np.linspace(0, 1002, 1003)


print(f"df_all = ")
print(df_all)

line_cand = ['solid', 'dashdot', 'dashed', 'dotted', 'solid']

# Function to create a list 

# for vx, ex, wz, ew in df_all.groupby(['cmd_vel']):
#     for 
plt.plot(plt_x,df_all['cmd_vel_x'], color = 'r', linestyle = 'solid', label='cmd_vel_x')
plt.plot(plt_x,df_all['base_vel_x'], color = 'r', linestyle = 'dashed', label=f'base_vel_x err={round(df_all["err_vel_x"].mean(),3)}')
# vel_x_err_mean = df_all['err_vel_x'].mean()
# print(f"vel_x_err_mean = {vel_x_err_mean}")
plt.plot(plt_x,df_all['cmd_vel_yaw'], color = 'k', linestyle = 'solid', label='cmd_vel_yaw')
plt.plot(plt_x,df_all['base_vel_yaw'], color = 'k', linestyle = 'dashed', label=f'base_vel_yaw, err={round(df_all["err_vel_yaw"].mean(),3)}')


# plt.xlim([0, cot_vel_max_gym])
# plt.ylim([0, 250])
plt.xlabel('Time [step]')
plt.ylabel('Command velocity [m/s]')
# ax2.set_ylabel('CoT [J/kg/m]')
plt.legend(loc='best')
plt.show()

