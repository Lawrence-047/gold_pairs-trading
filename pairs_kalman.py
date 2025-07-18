#库函数
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial Unicode MS'  
# plt.rcParams['font.family'] = 'SimHei'    
plt.rcParams['axes.unicode_minus'] = False  

def combine_md_datetime(df, date_col='MDDate', time_col='MDTime', new_col='DateTime', drop_original=True):
    """
    时间标准化，因为这边用的数据输入的时间是两列，
    通用型的时候可以增加datetime鉴别
    """
    df[date_col] = df[date_col].astype(str)
    df[time_col] = df[time_col].astype(str).str.zfill(9)
    datetime_str = df[date_col] + ' ' + df[time_col].str[:6]
    df[new_col] = pd.to_datetime(datetime_str, format='%Y%m%d %H%M%S')
    if drop_original:
        df.drop(columns=[date_col, time_col], inplace=True)
    return df

def load_and_prepare(filepath, prefix):
    """
    加载数据，清洗数据，加前缀
    """
    df = pd.read_csv(filepath, usecols=['MDDate', 'MDTime', 'LastPx', 'Buy1Price', 'Sell1Price'])
    df = df[df['LastPx'] != 0]
    df = combine_md_datetime(df)
    
    df.rename(columns={
        'LastPx': f'{prefix}_Price',
        'Buy1Price': f'{prefix}_Buy1Price',
        'Sell1Price': f'{prefix}_Sell1Price'
    }, inplace=True)
    
    return df


def merge_and_aggregate(df1, df2, group_size=30):
    """
    合并聚合生成merged
    """
    merged = pd.merge_asof(df1,df2,on='DateTime', direction='backward')
    
    merged['group'] = merged.index // group_size
    merged_agg = merged.groupby('group').tail(1).reset_index(drop=True)
    merged_agg.drop(columns=['group'], inplace=True)
    
    return merged_agg

#Kalman参数
OBS_VAR = 0.01
STATE_VAR = 0.0001
INIT_MEAN = 1.0
INIT_VAR = 1.0

def kalman_and_cointegration(df, win=60):
    """
    Kalman滤波，协整检验
    这个window可以调整，因为最后准备做一个通用型的，
    所以考虑之后加一个鉴别数据频率的函数，根据函数生成所需要的窗口大小
    """
    #识别价格列
    price_cols = [col for col in df.columns if col != 'DateTime' and pd.api.types.is_numeric_dtype(df[col])]
    if len(price_cols) < 2:
        raise ValueError("找不到两个价格列，至少需要两列数值型价格")

    col_y, col_x = price_cols[:2]

    #Kalman估计beta（for循环）
    """
    感觉这边是可以优化的，不然迭代速率有点低
    """
    state_mean = INIT_MEAN
    state_var = INIT_VAR
    betas, spreads = [], []

    for i in range(len(df)):
        py = df.loc[i, col_y]
        px = df.loc[i, col_x]

        pred_mean = state_mean
        pred_var = state_var + STATE_VAR
        k_gain = pred_var / (pred_var * px**2 + OBS_VAR)
        state_mean = pred_mean + k_gain * (py - pred_mean * px)
        state_var = (1 - k_gain * px) * pred_var

        beta = state_mean
        spread = py - beta * px

        betas.append(beta)
        spreads.append(spread)

    df['beta_kalman'] = betas
    df['spread'] = spreads
    df['spread_mean'] = pd.Series(spreads).rolling(win).mean()
    df['spread_std'] = pd.Series(spreads).rolling(win).std()
    df['zscore'] = (df['spread'] - df['spread_mean']) / df['spread_std']

    #静态协整检验
    """
    其实理论上应该是先协整检验，然后动态估计beta，不然会造成计算冗余
    但准备等参数调好再进行
    """
    X = sm.add_constant(df[col_x])
    model = sm.OLS(df[col_y], X).fit()
    beta_static = model.params[1]
    residuals = model.resid
    adf_stat, p_value, _, _, crit_vals, _ = adfuller(residuals)

    test_result = {
        'price_y': col_y,
        'price_x': col_x,
        '静态beta': beta_static,
        'ADF统计量': adf_stat,
        'p值': p_value,
        '5%临界值': crit_vals['5%'],
        '是否协整(5%)': p_value < 0.05
    }

    #画图看看
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(df['DateTime'], df['beta_kalman'], label='Kalman Beta')
    axes[0].set_title("动态β（Kalman估计）")
    axes[0].legend()
    axes[0].grid()

    axes[1].plot(df['DateTime'], df['spread'], label='Spread (残差)', color='orange')
    axes[1].set_title("残差 Spread")
    axes[1].legend()
    axes[1].grid()

    axes[2].plot(df['DateTime'], df['zscore'], label='Z-score of Spread', color='green')
    axes[2].axhline(2, linestyle='--', color='gray')
    axes[2].axhline(-2, linestyle='--', color='gray')
    axes[2].set_title("Z-score标准化的残差")
    axes[2].legend()
    axes[2].grid()

    plt.tight_layout()
    plt.show()

    #对比一下静态和动态beta，但这个好像没啥价值，我自己的代码里面已经改成了预估的beta和实际beta的对比图。
    plt.figure(figsize=(14, 4))
    plt.plot(df['DateTime'], df['beta_kalman'], label='动态 β (Kalman)', color='blue')
    plt.axhline(beta_static, linestyle='--', color='red', label=f'静态 β (OLS): {beta_static:.4f}')
    plt.title("动态 β vs. 静态 β 对比")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    return df[['DateTime', col_y, col_x, 'beta_kalman', 'spread', 'zscore']], test_result
