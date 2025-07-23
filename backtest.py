import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial Unicode MS'  
# plt.rcParams['font.family'] = 'SimHei'    
plt.rcParams['axes.unicode_minus'] = False  


def calculate_pair_strategy_returns(
    df, 
    price_y='ETF_Price', 
    price_x='TD_Price', 
    beta_col='beta_kalman', 
    signal_col='signal',
    time_col='DateTime',
    freeze_beta=True,
):
    """
    计算配对策略组合收益，并绘制持仓图与收益图
    开仓时冻结 beta，持仓期保持不变
    """
    df = df.copy()

    #生成 position
    df['position'] = df[signal_col].shift(1).fillna(0)

    #冻结 beta
    if freeze_beta:
        frozen_beta = []
        current_beta = None
        for i in range(len(df)):
            sig = df.iloc[i][signal_col]
            if sig == 1 or sig == -1:
                if current_beta is None:
                    current_beta = df.iloc[i][beta_col]
                frozen_beta.append(current_beta)
            else:
                current_beta = None
                frozen_beta.append(None)
        df['frozen_beta'] = frozen_beta
        df['frozen_beta'] = df['frozen_beta'].fillna(method='ffill')
        beta_used = 'frozen_beta'
    else:
        beta_used = beta_col

    #计算收益
    df['ret_y'] = df[price_y].pct_change().fillna(0)
    df['ret_x'] = df[price_x].pct_change().fillna(0)
    df['combo_ret'] = df['position'] * (df['ret_y'] - df[beta_used] * df['ret_x'])

    #累计收益
    df['cum_ret'] = (1 + df['combo_ret']).cumprod()
    df['cum_return'] = df['combo_ret'].cumsum()  # 可选：从0开始的累计收益

    # 画图
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    #持仓
    axes[0].plot(df[time_col], df['position'], label='Position', color='orange')
    axes[0].set_title("持仓信号")
    axes[0].set_ylim([-1.1, 1.1])
    axes[0].grid()
    axes[0].legend()

    # 净值收益图
    axes[1].plot(df[time_col], df['cum_ret'], label='Net Value (cum_ret)', color='green')
    # axes[1].plot(df[time_col], df['cum_return'], label='Cumulative Return (from 0)', color='blue', linestyle='--')
    axes[1].set_title("策略累计收益")
    axes[1].grid()
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()

    return df

def plot_beta_vs_frozen_with_signals(
    df,
    beta_col='beta_kalman',
    frozen_col='frozen_beta',
    signal_col='signal',
    time_col='DateTime'):
    """
    动态beta与冻结beta
    """
    df = df.copy()

    # 计算交易信号点
    entry_long = df[(df[signal_col] == 1) & (df[signal_col].shift(1) == 0)]
    entry_short = df[(df[signal_col] == -1) & (df[signal_col].shift(1) == 0)]
    exit = df[(df[signal_col] == 0) & (df[signal_col].shift(1) != 0)]

    plt.figure(figsize=(14, 6))
    
    # beta线条
    plt.plot(df[time_col], df[beta_col], label='Kalman β (Dynamic)', color='blue', linewidth=1.5)
    plt.plot(df[time_col], df[frozen_col], label='Frozen β (Trading)', color='orange', linestyle='--', linewidth=1.5)

    # 交易信号点
    plt.scatter(entry_long[time_col], entry_long[frozen_col], marker='^', color='green', label='Long Entry', zorder=5)
    plt.scatter(entry_short[time_col], entry_short[frozen_col], marker='v', color='red', label='Short Entry', zorder=5)

    # 图设置
    plt.title("动态 β vs 冻结 β 与交易信号")
    plt.xlabel("时间")
    plt.ylabel("β")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()