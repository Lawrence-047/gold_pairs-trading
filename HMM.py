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
    state_col='beta_state',
    n_init=10
):
    """
    对beta波动率进行建模
    HMM regime拟合
    """
    df = df.copy()

    ##计算beta_vol，然后log和标准化
    df[vol_col] = df[beta_col].rolling(window=window, min_periods=10).std()
    df[vol_col] = np.log(df[vol_col] + 1e-9)
    df = df.dropna(subset=[vol_col, zscore_col])

    #组合变量
    obs = np.column_stack([df[vol_col].values, df[zscore_col].values])
    scaler = StandardScaler()
    obs_scaled = scaler.fit_transform(obs)
    
    # === Step 2: 多次初始化 HMM，选 log likelihood 最大的 ===
    best_model = None
    best_score = -np.inf
    best_states = None

    for seed in range(n_init):
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=1000,
            random_state=seed,
            init_params='stmc'
        )
        try:
            model.fit(obs_scaled)
            score = model.score(obs_scaled)
            hidden_states = model.predict(obs_scaled)

            if score > best_score:
                best_score = score
                best_model = model
                best_states = hidden_states
        except Exception as e:
            print(f"Seed {seed} failed: {e}")
            continue

    # === Step 3: 状态重新映射（按 beta_vol 均值排序）===
    beta_vol_values = obs_scaled[:, 0]
    means = [beta_vol_values[best_states == i].mean() for i in range(n_states)]
    state_order = np.argsort(means)
    mapping = {original: new for new, original in enumerate(state_order)}
    mapped_states = np.array([mapping[s] for s in best_states])
    df[state_col] = mapped_states
    
    # === Step 4: 画图 ===
    times = df[time_col]
    beta_vol_plot = df[vol_col]
    states = df[state_col]

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(times, beta_vol_plot, label='Beta Volatility (log)', color='black', linewidth=1.2)
    ax1.set_ylabel('Log Beta Volatility (black)')
    ax1.set_xlabel('Time')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.step(times, states, label='HMM State (0=Low, 1=Mid, 2=High)', color='red', linewidth=1.2, where='post', alpha=0.6)
    ax2.set_ylabel('HMM State (red)')
    ax2.set_ylim(-0.1, n_states - 0.9)
    ax2.set_yticks(range(n_states))
    ax2.set_yticklabels(['Low', 'Mid', 'High'])

    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.xticks(rotation=30)
    plt.title('Beta Volatility (log) & Z-score with HMM Regime State')
    fig.tight_layout()
    plt.show()

    return df, best_model

def generate_signal_with_hmm_state(
    df,
    z_col='zscore',
    beta_col='beta_kalman',
    state_col='beta_state',
    time_col='DateTime',
    close_h=0.5,
    close_l=-0.5,
    stop_spread=1.5,
    max_hold_minutes=60,
    start_index=None,
    state_open_thresholds={0: (2.5, -2.5), 1: (3.0, -3.0), 2: (3.5, -3.5)}  # 低-中-高波动门槛
):
    df = df.copy()
    n = len(df)
    df['signal'] = 0
    df['frozen_beta'] = np.nan

    position = 0
    entry_z = None
    entry_time = None
    frozen_beta = None

    start_i = start_index if start_index is not None else 0

    for i in range(start_i, n):
        z = df.at[df.index[i], z_col]
        t = df.at[df.index[i], time_col]
        beta_now = df.at[df.index[i], beta_col]
        state = df.at[df.index[i], state_col]

        # 获取当前状态下的开仓门槛
        if np.isnan(state):
            open_h, open_l = 3.0, -3.0  # fallback
        else:
            open_h, open_l = state_open_thresholds.get(int(state), (3.0, -3.0))

        hold_minutes = (t - entry_time).total_seconds() / 60 if entry_time else 0

        if position == 0:
            if z < open_l:
                position = 1
                entry_z = z
                entry_time = t
                frozen_beta = beta_now if np.isfinite(beta_now) else 1.0
                df.at[df.index[i], 'signal'] = 1
            elif z > open_h:
                position = -1
                entry_z = z
                entry_time = t
                frozen_beta = beta_now if np.isfinite(beta_now) else 1.0
                df.at[df.index[i], 'signal'] = -1
            else:
                frozen_beta = beta_now if np.isfinite(beta_now) else frozen_beta
                df.at[df.index[i], 'signal'] = 0
        else:
            profit_cond = (close_l < z < close_h)
            stop_cond = ((position == 1 and z < entry_z - stop_spread) or
                         (position == -1 and z > entry_z + stop_spread))
            time_cond = hold_minutes >= max_hold_minutes

            if profit_cond or stop_cond or time_cond:
                frozen_beta = beta_now if np.isfinite(beta_now) else frozen_beta
                df.at[df.index[i], 'signal'] = 0
                position = 0
                entry_z = None
                entry_time = None
            else:
                df.at[df.index[i], 'signal'] = position

        df.at[df.index[i], 'frozen_beta'] = frozen_beta

    return df