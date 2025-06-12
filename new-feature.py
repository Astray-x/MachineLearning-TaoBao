import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


CONFIG = {
    'input_file': 'cleaned_UserBehavior_1M.csv',
    'output_path': 'high_value_conversion_features_final.csv',
    'high_value_actions': ['pv', 'fav', 'cart'],
    'time_windows': ['1h', '24h', '72h'],
    'min_user_actions': 3,
    'behavior_pairs': [
        ('pv', 'fav'),
        ('pv', 'cart'),
        ('fav', 'cart')
    ]
}


def load_data(filepath):
    #数据加载与基础预处理

    try:
        df = pd.read_csv(filepath, parse_dates=['datetime'])
        df = df.sort_values(['user_id', 'datetime'])

        user_action_counts = df.groupby('user_id').size()
        valid_users = user_action_counts[user_action_counts >= CONFIG['min_user_actions']].index
        df = df[df['user_id'].isin(valid_users)]

        print(f"成功加载数据，有效记录数: {len(df):,}")
        return df
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        raise


def create_time_features(df):
    #创建时间特征

    minutes_in_day = 24 * 60
    df['minute_sin'] = np.sin(2 * np.pi * (df['datetime'].dt.hour * 60 + df['datetime'].dt.minute) / minutes_in_day)
    df['minute_cos'] = np.cos(2 * np.pi * (df['datetime'].dt.hour * 60 + df['datetime'].dt.minute) / minutes_in_day)

    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_peak_hour'] = df['datetime'].dt.hour.isin([10, 11, 15, 16, 20, 21]).astype(int)
    return df


def calculate_behavior_sequences(df):
    #计算行为序列特征

    action_map = {'pv': 1, 'fav': 2, 'cart': 3, 'buy': 4}
    df['action_code'] = df['behavior_type'].map(action_map)

    for start, end in CONFIG['behavior_pairs']:
        df[f'{start}_to_{end}'] = (
                (df['behavior_type'] == start) &
                (df['action_code'].shift(-1) == action_map[end]) &
                (df['user_id'] == df['user_id'].shift(-1))
        ).astype(int)

        df[f'{start}_to_{end}_time'] = df.groupby('user_id')['datetime'].diff(-1).dt.total_seconds().where(
            df[f'{start}_to_{end}'] == 1
        )

        print(f"生成转化路径: {start}=>{end}")

    return df


def create_window_features(df):
    #创建窗口统计特征

    # 计算用户平均行为间隔
    user_freq = df.groupby('user_id')['datetime'].diff().dt.total_seconds().groupby(df['user_id']).mean()
    df['user_avg_interval'] = df['user_id'].map(user_freq)

    # 高价值行为标记
    df['is_high_value'] = df['behavior_type'].isin(CONFIG['high_value_actions']).astype(int)

    # 固定窗口大小（基于数据采样频率）
    base_window_size = max(10, int(3600 / df['user_avg_interval'].median()))
    window_sizes = {
        '1h': base_window_size,
        '24h': base_window_size * 24,
        '72h': base_window_size * 72
    }

    for window in CONFIG['time_windows']:
        window_size = max(5, min(1000, window_sizes[window]))  # 限制在5-1000之间

        # 高价值行为比率
        df[f'{window}_high_ratio'] = df.groupby('user_id')['is_high_value'].transform(
            lambda x: x.rolling(window=window_size, min_periods=1).mean()
        )

        # 转化路径统计
        for start, end in CONFIG['behavior_pairs']:
            df[f'{window}_{start}_to_{end}_rate'] = df.groupby('user_id')[f'{start}_to_{end}'].transform(
                lambda x: x.rolling(window=window_size, min_periods=1).mean()
            )

    return df


def create_user_profiles(df):

    # 定义聚合函数
    agg_funcs = {
        'behavior_type': [
            ('total_actions', 'count'),
            ('pv_count', lambda x: (x == 'pv').sum()),
            ('fav_count', lambda x: (x == 'fav').sum()),
            ('cart_count', lambda x: (x == 'cart').sum())
        ]
    }

    # 添加转化路径统计
    for start, end in CONFIG['behavior_pairs']:
        agg_funcs[f'{start}_to_{end}'] = 'sum'
        agg_funcs[f'{start}_to_{end}_time'] = 'mean'

    user_stats = df.groupby('user_id').agg(agg_funcs)

    user_stats.columns = ['_'.join(col).strip() for col in user_stats.columns.values]

    # 计算关键转化率
    for start, end in CONFIG['behavior_pairs']:
        user_stats[f'{start}_to_{end}_rate'] = (
                user_stats[f'{start}_to_{end}_sum'] /
                (user_stats['behavior_type_pv_count'] + 1e-6)
        )


    df = df.merge(user_stats, on='user_id', suffixes=('', '_y'))

    # 移除重复列
    dup_cols = [col for col in df.columns if col.endswith('_y')]
    df = df.drop(dup_cols, axis=1)

    return df


def create_interaction_features(df):
    #创建交互特征

    if 'behavior_type_total_actions' in df.columns:
        total_col = 'behavior_type_total_actions'
    else:
        total_col = 'total_actions'

    # 行为密度特征
    df['pv_density'] = df['behavior_type_pv_count'] / (df[total_col] + 1)
    df['cart_intensity'] = df['behavior_type_cart_count'] * df['24h_high_ratio']

    # 时间交互特征
    df['weekend_cart_boost'] = df['is_weekend'] * df['pv_to_cart_rate']
    df['peak_hour_fav_boost'] = df['is_peak_hour'] * df['pv_to_fav_rate']

    # 转化速度特征
    for start, end in CONFIG['behavior_pairs']:
        time_col = f'{start}_to_{end}_time_mean'
        if time_col in df.columns:
            df[f'{start}_to_{end}_speed'] = 1 / (df[time_col] + 1e-6)

    return df


def generate_target(df):

    df['next_high_value'] = (
            (df['behavior_type'].shift(-1).isin(CONFIG['high_value_actions'])) &
            (df['user_id'] == df['user_id'].shift(-1))
    ).astype(int)

    df['next_is_conversion'] = (
            (df['behavior_type'].shift(-1) == 'cart') &
            (df['user_id'] == df['user_id'].shift(-1))
    ).astype(int)

    pos_ratio = df['next_high_value'].mean()
    print(f"目标变量分布 - 高价值行为: {pos_ratio:.2%}")
    print(f"目标变量分布 - 加购转化: {df['next_is_conversion'].mean():.2%}")

    return df


def save_features(df):
    print("保存特征文件...")

    auto_drop = ['action_code', 'is_high_value', 'user_avg_interval'] + \
                [f'{start}_to_{end}_time' for start, end in CONFIG['behavior_pairs']]

    keep_cols = [col for col in df.columns if col not in auto_drop]
    final_df = df[keep_cols]

    # 类型转换优化
    for col in final_df.select_dtypes('float64'):
        final_df[col] = final_df[col].astype('float32')

    final_df.to_csv(CONFIG['output_path'], index=False)
    print(f"特征保存完成，最终维度: {final_df.shape}")
    print(f"生成特征数量: {len(final_df.columns)}")


def main():

    try:
        df = load_data(CONFIG['input_file'])
        df = create_time_features(df)
        df = calculate_behavior_sequences(df)
        df = create_window_features(df)
        df = create_user_profiles(df)
        df = create_interaction_features(df)
        df = generate_target(df)
        save_features(df)
        return df
    except Exception as e:
        print(f"流程执行失败: {str(e)}")
        raise


if __name__ == "__main__":
    feature_data = main()