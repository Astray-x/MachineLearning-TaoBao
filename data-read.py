import pandas as pd

# 加载数据
column_names = ['user_id', 'item_id', 'category_id', 'behavior_type', 'timestamp']
df = pd.read_csv(
    'UserBehavior.csv',
    header=None,
    names=column_names,
    nrows=1000000
)

# 删除重复值
df = df.drop_duplicates()

# 异常值处理（用户行为次数）
user_behavior_counts = df['user_id'].value_counts()
Q1, Q3 = user_behavior_counts.quantile([0.25, 0.75])
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
normal_users = user_behavior_counts[
    (user_behavior_counts >= lower_bound) &
    (user_behavior_counts <= upper_bound)
].index
df = df[df['user_id'].isin(normal_users)]

# 类型转换
df = df.astype({
    'user_id': 'int32',
    'item_id': 'int32',
    'category_id': 'int32',
    'timestamp': 'int64'
})
df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')

# 保存数据
df.to_csv('cleaned_UserBehavior_1M.csv', index=False)
print("数据已保存为 cleaned_UserBehavior_1M.csv")