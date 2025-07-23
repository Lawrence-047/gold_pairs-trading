import pandas as pd
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial Unicode MS'  
# plt.rcParams['font.family'] = 'SimHei'    
plt.rcParams['axes.unicode_minus'] = False

#Kalman参数
OBS_VAR = 0.01
STATE_VAR = 0.0001
INIT_VAR = 1.0

@njit
def kalman_filter_numba(py, px, obs_var, state_var, init_mean, init_var):
    n = len(py)
    betas = np.zeros(n)
    spreads = np.zeros(n)

    state_mean = init_mean
    state_var_ = init_var

    for i in range(n):
        pred_mean = state_mean
        pred_var = state_var_ + state_var
        k_gain = pred_var / (pred_var * px[i]**2 + obs_var)
        state_mean = pred_mean + k_gain * (py[i] - pred_mean * px[i])
        state_var_ = (1 - k_gain * px[i]) * pred_var

        betas[i] = state_mean
        spreads[i] = py[i] - state_mean * px[i]

    return betas, spreads

def kalman_and_cointegration(df, win=10,early_stop=True):
    """
    Kalman滤波，协整检验
    这个window可以调整，因为最后准备做一个通用型的，
    所以考虑之后加一个鉴别数据频率的函数，根据函数生成所需要的窗口大小
    """
    #识别价格列
    price_cols = [col for col in df.columns if col != 'DateTime' and pd.api.types.is_numeric_dtype(df[col])]
    if len(price_cols) < 2:
        raise ValueError("找不到两个价格列，至少需要两列价格")

    col_y, col_x = price_cols[:2]

    #Kalman估计beta（for循环）
    """
    这里用numba优化了一下，准备看看跑的情况
    """
    py = df[col_y].values.astype(np.float64)
    px = df[col_x].values.astype(np.float64)
    
    INIT_MEAN = py[0] / px[0]
    
    betas, spreads = kalman_filter_numba(
        py, px, OBS_VAR, STATE_VAR, INIT_MEAN, INIT_VAR
    )

    df['beta_kalman'] = betas
    df['spread'] = spreads
    lambda_ = 0.01  # 越小越平滑
    df['spread_mean'] = df['spread'].ewm(alpha=lambda_).mean()
    df['spread_std'] = df['spread'].ewm(alpha=lambda_).std()
    df['zscore'] = (df['spread'] - df['spread_mean']) / df['spread_std']

    #画图看看
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(df['DateTime'], df['zscore'], label='Z-score of Spread', color='green')
    axes[0].axhline(2, linestyle='--', color='gray')
    axes[0].axhline(-2, linestyle='--', color='gray')
    axes[0].set_title("Z-score标准化的残差")
    axes[0].legend()
    axes[0].grid()

    axes[1].plot(df['DateTime'], df['spread_std'], label='Spread_std', color='purple')
    axes[1].set_title("价差 Spread_std")
    axes[1].legend()
    axes[1].grid()
    
    plt.tight_layout()
    plt.show()
    
    return df[['DateTime', col_y, col_x, 'beta_kalman', 'spread', 'zscore']]