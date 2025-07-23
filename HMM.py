import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler

def train_hmm_on_beta_vol_and_plot(
    df,
    beta_col='beta_kalman',
    zscore_col='zscore',
    time_col='DateTime',
    window=60,
    n_states=3,
    vol_col='beta_vol',
    state_col='beta_state'
):
    """
    beta波动率进行建模
    HMM regime 拟合
    分了三个状态可视化
    """
    df = df.copy()

    #计算beta_vol，然后log和标准化
    df[vol_col] = df[beta_col].rolling(window=window, min_periods=10).std()
    df[vol_col] = np.log(df[vol_col] + 1e-9)
    df = df.dropna(subset=[vol_col, zscore_col])

    #组合变量
    obs = np.column_stack([df[vol_col].values, df[zscore_col].values])
    scaler = StandardScaler()
    obs_scaled = scaler.fit_transform(obs)
    
    #构建HMM，参数
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=1000,
        init_params="",  # 不再随机初始化 means, covars
        random_state=42
    )

    #初始状态概率
    model.startprob_ = np.array([1.0 / n_states] * n_states)
    #转移矩阵（这边主要考虑了一个状态的延续性，避免频繁变动）
    model.transmat_ = np.array([
        [0.94, 0.04, 0.02],
        [0.04, 0.92, 0.04],
        [0.02, 0.04, 0.94]
    ])

    #标准化后的初始状态均值
    model.means_ = np.array([
        [-1.5, -1.0],
        [0.0, 0.0],
        [1.5, 1.0]
    ])
    
    #协方差
    model.covars_ = np.tile(np.identity(2) * 2.0, (n_states, 1, 1))

    #拟合
    model.fit(obs_scaled)
    hidden_states = model.predict(obs_scaled)

    #排序状态
    vol_means = model.means_[:, 0]
    state_order = np.argsort(vol_means)
    mapping = {original: new for new, original in enumerate(state_order)}
    mapped_states = np.array([mapping[s] for s in hidden_states])
    df[state_col] = mapped_states

    #画几个图
    times = df[time_col]
    beta_vol_plot = df[vol_col]
    states = df[state_col]

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(times, beta_vol_plot, label='Beta Volatility (log)', color='black', linewidth=1.2)
    ax1.set_ylabel('Log Beta Volatility (black)')
    ax1.set_xlabel('Time')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.step(times, states, label='HMM State', color='red', linewidth=1.2, where='post', alpha=0.6)
    ax2.set_ylabel('HMM State (red)')
    ax2.set_ylim(-0.1, n_states - 0.9)
    ax2.set_yticks(range(n_states))
    ax2.set_yticklabels(['Low', 'Med', 'High'][:n_states])

    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.xticks(rotation=30)
    plt.title('Beta Volatility (log) and Z-score with 3-State HMM Regime')
    fig.tight_layout()
    plt.show()

    return df, model