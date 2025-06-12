import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
import warnings
import json
import sys

warnings.filterwarnings('ignore')

CONFIG = {
    'time_split_ratio': 0.8,  # 训练集测试集时间分割比例
    'window_sizes': ['1D', '3D', '7D'],  # 滑动窗口大小配置
    'top_categories': 20,  # 保留的高频品类数量
    'min_user_actions': 5  # 用户最小行为次数阈值
}


def load_and_preprocess(filepath):

    df = pd.read_csv(
        filepath,
        parse_dates=['datetime'],
        dtype={
            'user_id': 'int32',
            'item_id': 'int32',
            'category_id': 'int32',
            'behavior_type': 'category'
        }
    )

    # 基础数据清洗
    df = df.drop_duplicates()
    df = df.sort_values(['user_id', 'datetime'])

    # 过滤低活跃用户
    user_actions = df['user_id'].value_counts()
    active_users = user_actions[user_actions >= CONFIG['min_user_actions']].index
    df = df[df['user_id'].isin(active_users)]
    print(f"数据加载完成，有效用户数：{len(active_users)}")
    return df


def create_time_features(df, reference_date):

    df['days_since_ref'] = (reference_date - df['datetime']).dt.days
    df['hour_sin'] = np.sin(2 * np.pi * df['datetime'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['datetime'].dt.hour / 24)
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    bins = [0, 6, 12, 18, 24]
    labels = ['凌晨', '上午', '下午', '晚上']
    df['time_of_day'] = pd.cut(df['datetime'].dt.hour, bins=bins, labels=labels)
    return df


def create_behavior_features(df, train_mask):

    behavior_cols = []

    for window in CONFIG['window_sizes']:

        df.loc[train_mask, f'{window}_actions'] = df[train_mask].groupby('user_id')['behavior_type'] \
            .transform(lambda x: x.rolling(window, min_periods=1).count())

        for behavior in ['pv', 'cart', 'fav']:
            col_name = f'{window}_{behavior}_ratio'
            df.loc[train_mask, col_name] = df[train_mask].groupby('user_id')['behavior_type'] \
                .transform(lambda x: x.eq(behavior).rolling(window, min_periods=1).mean())
            behavior_cols.append(col_name)

        df.loc[train_mask, f'{window}_buy_prob'] = df[train_mask].groupby('user_id')['behavior_type'] \
            .transform(lambda x: x.eq('buy').rolling(window, min_periods=1).mean())

    # 测试集使用训练集统计量
    if (~train_mask).any():
        train_stats = df[train_mask].groupby('user_id')[behavior_cols].mean()
        for col in behavior_cols:
            df.loc[~train_mask, col] = df[~train_mask]['user_id'].map(train_stats[col])

    return df, behavior_cols


def create_sequence_features(df):

    df['prev_action'] = df.groupby('user_id')['behavior_type'].shift(1)
    df['next_action'] = df.groupby('user_id')['behavior_type'].shift(-1)
    df['action_pair'] = df['prev_action'] + '>' + df['behavior_type']
    df['time_since_last'] = df.groupby('user_id')['datetime'].diff().dt.total_seconds()
    df['time_to_next'] = df.groupby('user_id')['datetime'].diff(-1).dt.total_seconds().abs()
    df['session_id'] = (df.groupby('user_id')['time_since_last'] > 1800).cumsum()
    return df


def create_user_profiles(df, train_mask):

    if train_mask is not None:
        train_users = df.loc[train_mask, 'user_id'].unique()
        df = df[df['user_id'].isin(train_users)]

    user_stats = df.groupby('user_id').agg({
        'behavior_type': [
            ('total_actions', 'count'),
            ('pv_count', lambda x: x.eq('pv').sum()),
            ('cart_count', lambda x: x.eq('cart').sum()),
            ('buy_prob', lambda x: x.eq('buy').mean())
        ],
        'category_id': [
            ('unique_categories', 'nunique'),
            ('fav_category', lambda x: x.mode()[0] if len(x.mode()) > 0 else -1)
        ],
        'datetime': [
            ('first_seen', 'min'),
            ('last_seen', 'max'),
            ('activity_days', lambda x: x.nunique())
        ]
    })
    user_stats.columns = ['_'.join(col) for col in user_stats.columns]

    user_stats['actions_per_day'] = user_stats['behavior_type_total_actions'] / user_stats['datetime_activity_days']
    user_stats['purchase_rate'] = user_stats['behavior_type_buy_prob']

    return df.merge(user_stats, how='left', on='user_id')


def encode_categorical_features(df):

    top_categories = df['category_id'].value_counts().head(CONFIG['top_categories']).index
    df['category_group'] = np.where(
        df['category_id'].isin(top_categories),
        df['category_id'],
        'other'
    )

    cat_cols = ['behavior_type', 'prev_action', 'next_action', 'action_pair', 'time_of_day']
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    return df


def generate_target(df):

    df['target_action'] = df.groupby('user_id')['behavior_type'].shift(-1)
    df['future_buy'] = df.groupby('user_id')['behavior_type'] \
        .transform(lambda x: x.eq('buy').rolling('7D', min_periods=1).max().shift(-1))
    return df


def feature_selection(df):

    base_features = [
        'hour_sin', 'hour_cos', 'day_of_week', 'is_weekend', 'time_of_day',
        '1D_actions', '3D_pv_ratio', '7D_cart_ratio', 'time_since_last',
        'actions_per_day', 'purchase_rate', 'unique_categories',
        'action_pair', 'session_id'
    ]
    selected = [f for f in base_features if f in df.columns]
    return df[selected + ['user_id', 'datetime', 'target_action', 'future_buy']]


def save_features(df, output_path):

    # 保存特征数据
    df.to_csv(output_path, index=False)

    # 保存元数据
    meta = {
        'created_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'num_samples': len(df),
        'num_users': df['user_id'].nunique(),
        'feature_columns': [c for c in df.columns if c not in ['user_id', 'datetime']]
    }
    with open(output_path.replace('.csv', '_meta.json'), 'w') as f:
        json.dump(meta, f)
    print(f"特征数据已保存至 {output_path}")


def main(input_file, output_file):

    df = load_and_preprocess(input_file)

    split_date = df['datetime'].quantile(CONFIG['time_split_ratio'])
    train_mask = df['datetime'] <= split_date

    df = create_time_features(df, split_date)
    df, _ = create_behavior_features(df, train_mask)
    df = create_sequence_features(df)
    df = create_user_profiles(df, train_mask)
    df = encode_categorical_features(df)
    df = generate_target(df)
    final_df = feature_selection(df)

    save_features(final_df, output_file)
    return final_df


if __name__ == "__main__":
    input_path = sys.argv[1] if len(sys.argv) > 1 else 'cleaned_UserBehavior_1M.csv'
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'new_behavior_features.csv'

    features = main(input_path, output_path)