import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial Unicode MS'
plt.rcParams['axes.unicode_minus'] = False

def train_frozen_model_by_products(
    panel: pd.DataFrame,
    product_y: str,
    product_x: str,
    beta_cutoff: str = "2022-04-30",
    include_cutoff: bool = True,
    pre_days: int = 2,      # 先保留参数，但当前不做窗口扩展
    post_days: int = 0,     # 先保留参数，但当前不做窗口扩展
    obs_var: float = 0.01,
    state_var: float = 0.001,
    init_mean: float = 1.0,
    init_var: float = 1.0,
    std_floor: float = 1e-6,
    use_mad: bool = True,
    mad_scale: float = 1.4826,
    verbose: bool = True
):
    """
    用“生效主连”训练 β/μ/σ；剔除换月信号日与T+1、T+2（无窗口扩展）。
    """
    P_Y = product_y.upper()
    P_X = product_x.upper()
    date_col = "MDDate"

    # 1) 训练列：优先 *_main_eff，缺则回退 *_main
    y_col_eff = f"{P_Y}_main_eff"
    x_col_eff = f"{P_X}_main_eff"
    y_col_fallback = f"{P_Y}_main"
    x_col_fallback = f"{P_X}_main"

    y_col = y_col_eff if y_col_eff in panel.columns else y_col_fallback
    x_col = x_col_eff if x_col_eff in panel.columns else x_col_fallback

    for col in [y_col, x_col]:
        if col not in panel.columns:
            raise ValueError(f"[错误] 缺少列: {col}，请先生成 *_main_eff 或 *_main")

    if verbose and (y_col != y_col_eff or x_col != x_col_eff):
        print(f"[提示] 使用回退列：y={y_col}, x={x_col}")

    # 2) 构造训练掩码：剔除 signal_T / is_T1 / is_T2（不扩窗）
    def build_mask(panel: pd.DataFrame, products: list[str]) -> pd.DataFrame:
        df = panel.copy().sort_values(date_col).reset_index(drop=True)
        full_mask = pd.Series(True, index=df.index)

        for p in products:
            sig_col = f"{p.upper()}_signal_T"
            t1_col  = f"{p.upper()}_is_T1"
            t2_col  = f"{p.upper()}_is_T2"

            # 兼容性：不存在则当作 False
            roll_flag = (
                df.get(sig_col, pd.Series(False, index=df.index)).fillna(False).astype(bool) |
                df.get(t1_col,  pd.Series(False, index=df.index)).fillna(False).astype(bool) |
                df.get(t2_col,  pd.Series(False, index=df.index)).fillna(False).astype(bool)
            )

            full_mask &= ~roll_flag

        df["joint_train_mask"] = full_mask
        return df

    df = build_mask(panel, [P_Y, P_X])
    df[date_col] = pd.to_datetime(df[date_col])
    cutoff_ts = pd.to_datetime(beta_cutoff)

    if include_cutoff:
        df_train = df[(df[date_col] <= cutoff_ts) & df["joint_train_mask"]]
    else:
        df_train = df[(df[date_col] <  cutoff_ts) & df["joint_train_mask"]]

    df_train = df_train.dropna(subset=[y_col, x_col]).sort_values(date_col)

    if len(df_train) < 5:
        raise ValueError(f"[训练失败] 可用训练样本仅 {len(df_train)} 行，无法估计 beta")

    # 3) Kalman 估计 β（状态=常漂移；观测= y ≈ β x）
    yv = df_train[y_col].to_numpy()
    xv = df_train[x_col].to_numpy()
    mean, var = init_mean, init_var
    beta_path = []
    for y, x in zip(yv, xv):
        pred_mean = mean
        pred_var  = var + state_var
        k_gain    = pred_var * x / (pred_var * x**2 + obs_var)
        mean      = pred_mean + k_gain * (y - pred_mean * x)
        var       = (1 - k_gain * x) * pred_var
        beta_path.append(mean)

    frozen_beta = float(np.median(beta_path))

    # 4) 残差与标准化参数
    df_train["residual"] = df_train[y_col] - frozen_beta * df_train[x_col]
    residuals = df_train["residual"].to_numpy()

    mu = float(np.median(residuals)) if use_mad else float(np.mean(residuals))
    if use_mad:
        mad = float(np.median(np.abs(residuals - mu)))
        sigma = max(mad * mad_scale, std_floor)
    else:
        std = float(np.std(residuals, ddof=1))
        sigma = max(std, std_floor)

    if verbose:
        print(f"[√] β = {frozen_beta:.6f} | μ = {mu:.6f} | σ = {sigma:.6f} | 样本数 = {len(df_train)}")

    # 可选：残差图
    plt.figure(figsize=(12, 4))
    plt.plot(df_train[date_col], df_train["residual"], label="残差", linewidth=1.2)
    plt.axhline(mu, color="gray", linestyle="--", linewidth=1.0, label=f"μ = {mu:.4f}")
    plt.axhline(mu + 2*sigma, color="red", linestyle="--", linewidth=0.9, label=f"μ ± 2σ")
    plt.axhline(mu - 2*sigma, color="red", linestyle="--", linewidth=0.9)
    plt.title(f"残差序列（{product_y}/{product_x}，固定 β）", fontsize=14)
    plt.xlabel("日期"); plt.ylabel("残差")
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

    return {
        "frozen_beta": frozen_beta,
        "z_mu": mu,
        "z_std": sigma,
        "beta_path": np.array(beta_path),
        "train_df": df_train.reset_index(drop=True),
        "train_dates": df_train[date_col].tolist()
    }