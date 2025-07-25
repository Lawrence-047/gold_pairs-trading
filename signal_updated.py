#库函数
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from itertools import combinations
import numpy as np
from numba import njit

def generate_signal_with_weighted_z(
    df,
    z_col='zscore',
    beta_col='beta_kalman',
    time_col='DateTime',
    base_open_h=2.0,
    base_open_l=-2.0,
    alpha=5.0,
    close_h=0.5,
    close_l=-0.5,
    stop_spread=1.5,
    max_hold_minutes=15,
    beta_diff_base_tol=0.01,
    beta_diff_alpha_z=0.01,
    z_weight_gamma=3.0,
    start_index=None
):
    """
    有仓位时候冻结beta
    基于beta的动态偏离容忍区间增加平仓逻辑
    zscore会根据beta的偏离程度调整，增加开仓逻辑

    里面有几个参数：
    base_open_h,base_open_l: 基础开仓门槛
    alpha: beta_diff 越大，门槛越高（增强过滤）
    beta_diff_base_tol: beta_diff 基础容忍比例
    beta_diff_alpha_z: 容忍区间，影响平仓
    z_weight_gamma: beta_diff压缩z
    """
    df = df.copy()
    n = len(df)
    df['signal'] = 0
    df['frozen_beta'] = np.nan
    df['z_adj'] = np.nan

    position = 0
    entry_z = None
    entry_time = None
    frozen_beta = None
    start_i = start_index if start_index is not None else 0
    
    for i in range(start_i, n):
        z = df.at[df.index[i], z_col]
        t = df.at[df.index[i], time_col]
        beta_now = df.at[df.index[i], beta_col]

        #beta移动比例
        beta_diff_ratio = abs(beta_now - frozen_beta) / frozen_beta if frozen_beta else 0

        #设置容忍区间，随zscore
        beta_tol = beta_diff_base_tol + beta_diff_alpha_z * abs(z)

        #修正zscore权重
        z_adj = z / (1 + z_weight_gamma * beta_diff_ratio) if frozen_beta else z
        df.at[df.index[i], 'z_adj'] = z_adj

        #调整后的开仓门槛
        open_h = base_open_h + alpha * beta_diff_ratio
        open_l = base_open_l - alpha * beta_diff_ratio

        #持仓时长（平仓逻辑之一）
        hold_minutes = (t - entry_time).total_seconds() / 60 if entry_time else 0
        
        if position == 0:
            if z_adj < open_l:
                position = 1
                entry_z = z_adj
                entry_time = t
                frozen_beta = beta_now if np.isfinite(beta_now) else 1.0
                df.at[df.index[i], 'signal'] = 1
            elif z_adj > open_h:
                position = -1
                entry_z = z_adj
                entry_time = t
                frozen_beta = beta_now if np.isfinite(beta_now) else 1.0
                df.at[df.index[i], 'signal'] = -1
            else:
                #空仓刷新beta
                frozen_beta = beta_now if np.isfinite(beta_now) else frozen_beta
                df.at[df.index[i], 'signal'] = 0
        else:
            #平仓条件
            profit_cond = (close_l < z_adj < close_h)
            stop_cond = ((position == 1 and z_adj < entry_z - stop_spread) or
                         (position == -1 and z_adj > entry_z + stop_spread))
            time_cond = hold_minutes >= max_hold_minutes
            beta_drift_cond = beta_diff_ratio > beta_tol

            if profit_cond or stop_cond or time_cond or beta_drift_cond:
                frozen_beta = beta_now if np.isfinite(beta_now) else frozen_beta
                df.at[df.index[i], 'signal'] = 0
                position = 0
                entry_z = None
                entry_time = None
            else:
                df.at[df.index[i], 'signal'] = position

        #frozen_beta
        df.at[df.index[i], 'frozen_beta'] = frozen_beta

    return df