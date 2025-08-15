import pandas as pd
import numpy as np
import re
from typing import List, Literal

_DROP_COLS = [
    "MDTime", "tradingday", "SecurityType", "SecuritySubType",
    "SecurityIDSource", "HTSCSecurityID", "Symbol", "PeriodType", "NumTrades", "MacroDetail"
]


#数据清洗

def _load_and_clean(file_path: str,
                    date_col: str="MDDate",
                    id_col: str="SecurityID",
                    vol_col: str="TotalVolumeTrade") -> pd.DataFrame:
    dce = pd.read_parquet(file_path)
    
    #只保留必要列
    base_keep = {date_col, id_col, vol_col, "ClosePx", "SettlePrice"}
    keep_cols = [c for c in dce.columns if (c in base_keep) or (c not in _DROP_COLS)]
    dce = dce[keep_cols].copy()

    #标准化字段
    dce[date_col] = pd.to_datetime(dce[date_col], format="%Y%m%d", errors="coerce")
    dce[id_col]   = dce[id_col].astype(str)
    dce[vol_col]  = pd.to_numeric(dce[vol_col], errors="coerce")

    #解析品种与合约年月
    #允许品种大小写
    dce["product"] = dce[id_col].str.extract(r'^([A-Za-z]+)\d{4}$')[0].str.upper()
    ym = dce[id_col].str.extract(r'^[A-Za-z]+(?P<ym>\d{4})$')["ym"]
    dce["yy"]   = pd.to_numeric(ym.str[:2], errors="coerce")
    dce["mm"]   = pd.to_numeric(ym.str[2:], errors="coerce")
    dce["year"] = 2000 + dce["yy"]
    dce["month"]= dce["mm"]

    #清洗无效行并排序
    dce = dce.dropna(subset=[date_col, "product", id_col]).sort_values([date_col, id_col]).reset_index(drop=True)
    return dce

def _daily_top2(df: pd.DataFrame,
                date_col: str, id_col: str, vol_col: str, price_col: str) -> pd.DataFrame:
    """
    返回每日按成交量排序的前二合约（含价格/成交量），便于主/次主力选取。
    """
    #取每个(date, id)最后一条（避免同一天重复）
    day_rows = df[[date_col, id_col, vol_col, price_col]].drop_duplicates(subset=[date_col, id_col], keep="last")
    #每日按成交量取前二
    top2 = day_rows.sort_values([date_col, vol_col], ascending=[True, False])
    #为了后续处理方便，标注每日前两名排序序号
    top2["rank"] = top2.groupby(date_col)[vol_col].rank(method="first", ascending=False)
    return top2[top2["rank"] <= 2].copy()

#按照成交量最大和稳定确认拼接主连（输出主连和次主力信息）

def _stitch_by_volume_stable_with_next(prod_df: pd.DataFrame,
                                       product: str,
                                       date_col: str, id_col: str,
                                       vol_col: str, price_col: str,
                                       vol_ratio: float = 1.1,
                                       confirm_days: int = 2,
                                       min_stay_days: int = 0) -> pd.DataFrame:
    
    def _lookup_price_on_day(df, d, sec_id, price_col):
        if sec_id is None or pd.isna(sec_id): return np.nan
        loc = df[(df[date_col] == d) & (df[id_col] == sec_id)]
        if loc.empty: return np.nan
        return pd.to_numeric(loc.iloc[-1][price_col], errors="coerce")

    top2 = _daily_top2(prod_df, date_col, id_col, vol_col, price_col)

    second_map = (
        top2[top2["rank"] == 2]
        .set_index(date_col)[[id_col, price_col, vol_col]]
        .rename(columns={id_col: "next_id", price_col: "next_price", vol_col: "next_vol"})
    )

    first = (
        top2[top2["rank"] == 1]
        .sort_values(date_col)[[date_col, id_col, price_col, vol_col]]
        .rename(columns={id_col: "cand_id", price_col: "cand_price", vol_col: "cand_vol"})
        .reset_index(drop=True)
    )

    out_rows = []
    cur_id, cur_vol = None, None
    stay_days = 0
    confirm_streak = 0
    prev_main_id = None
    cond_prev = False  #上一日是否满足≥ratio*当前量：不存在未来数据影响

    for _, row in first.iterrows():
        d         = row[date_col]
        cand_id   = row["cand_id"]
        cand_vol  = row["cand_vol"]
        cand_px   = row["cand_price"]
        sec = second_map.loc[d] if d in second_map.index else pd.Series({"next_id":None,"next_price":np.nan,"next_vol":np.nan})

        switched_today = False
        signal_today = False  #信号日（T）

        if cur_id is None:
            cur_id, cur_vol = cand_id, cand_vol
            stay_days, confirm_streak = 1, 0
            switched_today = False
            cond_prev = False
        else:
            if cand_id != cur_id:
                #更新当前主力量为当日真实量
                today_main = prod_df[(prod_df[date_col] == d) & (prod_df[id_col] == cur_id)]
                if not today_main.empty:
                    cur_vol = pd.to_numeric(today_main.iloc[-1][vol_col], errors="coerce")

                cond_ratio = (pd.notna(cur_vol) and pd.notna(cand_vol) and (cand_vol >= vol_ratio * cur_vol))

                #信号日：首次从不满足换为满足比例条件
                if cond_ratio and (not cond_prev):
                    signal_today = True
                #维护上一日状态（仅在候选不等于当前时才延续；否则重置）
                cond_prev = cond_ratio

                #实际切换的判定
                if cond_ratio:
                    confirm_streak += 1
                else:
                    confirm_streak = 0

                if (confirm_streak >= confirm_days) and (stay_days >= min_stay_days):
                    prev_main_id = cur_id
                    cur_id, cur_vol = cand_id, cand_vol
                    stay_days, confirm_streak = 1, 0
                    switched_today = True
                else:
                    stay_days += 1
                    switched_today = False
            else:
                #候选=当前，继续持有，比例状态清零
                stay_days += 1
                confirm_streak = 0
                switched_today = False
                cond_prev = False
                if pd.notna(cand_vol):
                    cur_vol = cand_vol

        #今日主力价格
        if cur_id == cand_id:
            main_px, main_vol = cand_px, cand_vol
        else:
            main_px = _lookup_price_on_day(prod_df, d, cur_id, price_col)
            loc = prod_df[(prod_df[date_col] == d) & (prod_df[id_col] == cur_id)]
            main_vol = pd.to_numeric(loc.iloc[-1][vol_col], errors="coerce") if not loc.empty else np.nan

        #当日旧主力ID/价格
        if switched_today:
            old_id = prev_main_id
        else:
            old_id = out_rows[-1][f"{product}_main_id"] if len(out_rows) > 0 else None
        old_px = _lookup_price_on_day(prod_df, d, old_id, price_col)

        out_rows.append({
            date_col: d,
            f"{product}_main_id": cur_id,
            f"{product}_main": main_px,
            f"{product}_main_vol": main_vol,
            f"{product}_next_id": sec.get("next_id", None),
            f"{product}_next": sec.get("next_price", np.nan),
            f"{product}_next_vol": sec.get("next_vol", np.nan),
            f"{product}_old_id": old_id,
            f"{product}_old": old_px,
            f"{product}_switch_T": bool(switched_today),  #实际切换日
            f"{product}_signal_T": bool(signal_today),   
        })

    res = pd.DataFrame(out_rows).sort_values(date_col).reset_index(drop=True)

    # —— T+1 / T+2：基于 signal_T 的交易日偏移（跳过周末/节假日）
    is_T1 = np.zeros(len(res), dtype=bool)
    is_T2 = np.zeros(len(res), dtype=bool)
    sig = res[f"{product}_signal_T"].values
    for i, flag in enumerate(sig):
        if flag:
            if i+1 < len(res): is_T1[i+1] = True
            if i+2 < len(res): is_T2[i+2] = True
    res[f"{product}_is_T1"] = is_T1
    res[f"{product}_is_T2"] = is_T2
    
    # === 生效主连（T+1 仍旧，T+2 起切到新主连） ===
    eff_id = np.where(res[f"{product}_is_T1"].values,
                      res[f"{product}_old_id"].values,
                      res[f"{product}_main_id"].values)

    eff_px = []
    for d, use_old, oid, mid, old_px, main_px in zip(
            res[date_col],
            res[f"{product}_is_T1"].values,
            res[f"{product}_old_id"], res[f"{product}_main_id"],
            res[f"{product}_old"],    res[f"{product}_main"]):
        if use_old:
            px = old_px
            if pd.isna(px):  # 当日旧ID缺价→当日回表兜底，不前瞻
                px = _lookup_price_on_day(prod_df, d, oid, price_col)
        else:
            px = main_px
            if pd.isna(px):
                px = _lookup_price_on_day(prod_df, d, mid, price_col)
        eff_px.append(px)

    res[f"{product}_main_eff_id"] = eff_id
    res[f"{product}_main_eff"]    = eff_px
    return res

#159规则主连（主/次各自的目标合约），输出主/次+switch

def _target_ym_for_159(dt: pd.Timestamp) -> tuple:
    m = dt.month
    y = dt.year
    if 1 <= m <= 4:   # 1-4月 -> 当年05
        return (y, 5)
    elif 5 <= m <= 8: # 5-8月 -> 当年09
        return (y, 9)
    else:             # 9-12月 -> 次年01
        return (y + 1, 1)

def _stitch_by_159_with_next(prod_df: pd.DataFrame,
                             product: str,
                             date_col: str, id_col: str,
                             price_col: str, vol_col: str) -> pd.DataFrame:
    def _lookup_price_on_day(df, d, sec_id, price_col):
        if sec_id is None or pd.isna(sec_id): return np.nan
        loc = df[(df[date_col] == d) & (df[id_col] == sec_id)]
        if loc.empty: return np.nan
        return pd.to_numeric(loc.iloc[-1][price_col], errors="coerce")

    df = prod_df[[date_col, id_col, "year", "month", price_col, vol_col]].copy()
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df[vol_col]   = pd.to_numeric(df[vol_col], errors="coerce")

    dates = df[date_col].drop_duplicates().sort_values()
    tmap = pd.DataFrame({date_col: dates})
    tmap[["t_year", "t_month"]] = tmap[date_col].apply(lambda x: pd.Series(_target_ym_for_159(x)))
    tmap["t_yy"] = tmap["t_year"] % 100
    tmap["t_id"] = tmap["t_yy"].astype(int).map(lambda v: f"{v:02d}") + tmap["t_month"].astype(int).map(lambda v: f"{v:02d}")
    tmap["t_sec"] = product + tmap["t_id"]

    def _next_ym(row):
        if row["t_month"] == 5:   return (row["t_year"], 9)
        elif row["t_month"] == 9: return (row["t_year"] + 1, 1)
        else:                     return (row["t_year"], 5)

    tmap[["n_year", "n_month"]] = tmap.apply(_next_ym, axis=1, result_type="expand")
    tmap["n_yy"] = tmap["n_year"] % 100
    tmap["n_id"] = tmap["n_yy"].astype(int).map(lambda v: f"{v:02d}") + tmap["n_month"].astype(int).map(lambda v: f"{v:02d}")
    tmap["n_sec"] = product + tmap["n_id"]

    day_rows = df.drop_duplicates(subset=[date_col, id_col], keep="last")

    merged = tmap.merge(
        day_rows[[date_col, id_col, price_col, vol_col]],
        left_on=[date_col, "t_sec"], right_on=[date_col, id_col], how="left"
    ).rename(columns={id_col: f"{product}_main_id", price_col: f"{product}_main", vol_col: f"{product}_main_vol"})

    merged = merged.merge(
        day_rows[[date_col, id_col, price_col, vol_col]],
        left_on=[date_col, "n_sec"], right_on=[date_col, id_col], how="left"
    ).rename(columns={id_col: f"{product}_next_id", price_col: f"{product}_next", vol_col: f"{product}_next_vol"})

    merged = merged.sort_values(date_col).reset_index(drop=True)

    # —— 实际切换日：main_id 变化（第一行不算）
    main_id = merged[f"{product}_main_id"]
    switch_T = main_id.ne(main_id.shift(1)) & main_id.shift(1).notna()
    merged[f"{product}_switch_T"] = switch_T

    # —— 信号日：目标合约 t_sec 变化（第一行不算）
    signal_T = merged["t_sec"].ne(merged["t_sec"].shift(1)) & merged["t_sec"].shift(1).notna()
    merged[f"{product}_signal_T"] = signal_T

    # —— old_id / old：昨日主力ID 与 当日旧价
    old_id = main_id.shift(1)
    merged[f"{product}_old_id"] = old_id
    merged[f"{product}_old"] = [
        _lookup_price_on_day(prod_df, d, oid, price_col) for d, oid in zip(merged[date_col], old_id)
    ]

    # —— 交易日语义 T+1 / T+2：从 signal_T 推
    is_T1 = np.zeros(len(merged), dtype=bool)
    is_T2 = np.zeros(len(merged), dtype=bool)
    sig = merged[f"{product}_signal_T"].values
    for i, flag in enumerate(sig):
        if flag:
            if i+1 < len(merged): is_T1[i+1] = True
            if i+2 < len(merged): is_T2[i+2] = True
    merged[f"{product}_is_T1"] = is_T1
    merged[f"{product}_is_T2"] = is_T2

    # === 生效主连（T+1 仍用旧合约，T+2 起用新合约） ===
    eff_id = np.where(merged[f"{product}_is_T1"].values,
                      merged[f"{product}_old_id"].values,
                      merged[f"{product}_main_id"].values)

    eff_px = []
    for d, use_old, oid, mid, old_px, main_px in zip(
            merged[date_col],
            merged[f"{product}_is_T1"].values,
            merged[f"{product}_old_id"], merged[f"{product}_main_id"],
            merged[f"{product}_old"],    merged[f"{product}_main"]):
        if use_old:
            px = old_px
            if pd.isna(px):
                px = _lookup_price_on_day(prod_df, d, oid, price_col)
        else:
            px = main_px
            if pd.isna(px):
                px = _lookup_price_on_day(prod_df, d, mid, price_col)
        eff_px.append(px)

    merged[f"{product}_main_eff_id"] = eff_id
    merged[f"{product}_main_eff"]    = eff_px

    keep_cols = [
        date_col,
        f"{product}_main", f"{product}_main_id", f"{product}_main_vol",
        f"{product}_next", f"{product}_next_id", f"{product}_next_vol",
        f"{product}_old",  f"{product}_old_id",
        f"{product}_switch_T", f"{product}_signal_T",
        f"{product}_is_T1",    f"{product}_is_T2",
        f"{product}_main_eff", f"{product}_main_eff_id",
    ]
    return merged[keep_cols]

#生成

def build_single_product_main_panel(
    file_path: str,
    product: str,
    method: Literal["volume_stable", "159"],
    price_col: str = "ClosePx",
    date_col: str = "MDDate",
    id_col: str = "SecurityID",
    vol_col: str = "TotalVolumeTrade",
    vol_ratio: float = 1.2,
    confirm_days: int = 2,
    min_stay_days: int = 0,
) -> pd.DataFrame:
    df = _load_and_clean(file_path, date_col, id_col, vol_col)
    sub = df[df["product"] == product.upper()].copy()
    if sub.empty:
        raise ValueError(f"[错误] 文件 {file_path} 中未找到品种 {product}")

    if method == "volume_stable":
        return _stitch_by_volume_stable_with_next(
            sub, product.upper(), date_col, id_col, vol_col, price_col,
            vol_ratio=vol_ratio, confirm_days=confirm_days, min_stay_days=min_stay_days
        )
    elif method == "159":
        return _stitch_by_159_with_next(
            sub, product.upper(), date_col, id_col, price_col, vol_col
        )
    else:
        raise ValueError(f"未知拼接方法: {method}")
    
from typing import Dict, Any

def build_dual_product_panel(
    config_1: Dict[str, Any],
    config_2: Dict[str, Any],
    merge_on: str = "MDDate",
    how: Literal["outer", "inner"] = "outer"
) -> pd.DataFrame:
    """
    封装双标的拼接流程。输入两个标的的清洗/拼接参数，自动完成merge。
    config_1 / config_2 应包含字段：
        - file_path
        - product
        - method
        - price_col
        - 可选：vol_ratio / confirm_days / min_stay_days

    返回：
        合并后的主连+次主力面板，按日期对齐
    """
    def _build(config: Dict[str, Any]) -> pd.DataFrame:
        return build_single_product_main_panel(
            file_path=config["file_path"],
            product=config["product"],
            method=config["method"],
            price_col=config.get("price_col", "ClosePx"),
            vol_ratio=config.get("vol_ratio", 1.2),
            confirm_days=config.get("confirm_days", 2),
            min_stay_days=config.get("min_stay_days", 0)
        )

    panel_1 = _build(config_1)
    panel_2 = _build(config_2)

    merged = panel_1.merge(panel_2, on=merge_on, how=how)
    return merged.sort_values(merge_on).reset_index(drop=True)
