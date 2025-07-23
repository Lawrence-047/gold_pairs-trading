import pandas as pd

def combine_md_datetime(df, date_col='MDDate', time_col='MDTime', new_col='DateTime', drop_original=True):
    """
    时间标准化，因为输入的时间是两列，
    通用型的时候可以增加datetime鉴别
    """
    df[date_col] = df[date_col].astype(str)
    df[time_col] = df[time_col].astype(str).str.zfill(9)
    datetime_str = df[date_col] + ' ' + df[time_col].str[:6]
    df[new_col] = pd.to_datetime(datetime_str, format='%Y%m%d %H%M%S')
    if drop_original:
        df.drop(columns=[date_col, time_col], inplace=True)
    return df

def preprocess_and_aggregate_dual_session(df, freq='30S', method='last', time_col='DateTime'):
    """
    仅保留交易时段（09:30–11:30, 13:00–15:00），并按指定频率聚合
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df.set_index(time_col, inplace=True)

    session1 = df.between_time("09:30:00", "11:30:00")
    session2 = df.between_time("13:00:00", "15:00:00")
    df_filtered = pd.concat([session1, session2])

    if method == 'last':
        df_agg = df_filtered.resample(freq).last()
    elif method == 'mean':
        df_agg = df_filtered.resample(freq).mean()
    else:
        raise ValueError("Unsupported aggregation method")

    df_agg = df_agg.reset_index()
    return df_agg

def rename_price_columns(df, prefix, exclude_cols=['DateTime']):
    """
    给列名加前缀（排除 DateTime）
    """
    df = df.copy()
    df.columns = [f"{prefix}_{col}" if col not in exclude_cols else col for col in df.columns]
    return df

def load_and_process_md_file_from_path(filepath, prefix, freq='30S', method='last'):
    df = pd.read_csv(filepath)
    needed_cols = ['MDDate', 'MDTime', 'Buy1Price', 'Sell1Price']
    df = df[needed_cols].copy()
    df = combine_md_datetime(df)
    df = preprocess_and_aggregate_dual_session(df, freq=freq, method=method)
    df['Price'] = (df['Buy1Price'] + df['Sell1Price']) / 2
    df.drop(['Buy1Price', 'Sell1Price'], axis=1, inplace=True)
    df = rename_price_columns(df, prefix=prefix)
    return df

def merge_by_datetime(dfs):
    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on='DateTime', how='inner')
    return merged

def process_and_merge_all(file_prefix_list, freq='30S', method='last'):
    """
    参数:
        file_prefix_list:[(filepath1, prefix1), (filepath2, prefix2), ...]
        freq:聚合频率，默认30秒
        method:聚合方式，可选'last' or 'mean'
    返回:
        merged:清洗合并之后DataFrame
    """
    processed_dfs = []
    for path, prefix in file_prefix_list:
        df = load_and_process_md_file_from_path(path, prefix, freq=freq, method=method)
        processed_dfs.append(df)

    merged = merge_by_datetime(processed_dfs)
    merged.dropna(inplace=True)
    
    return merged

