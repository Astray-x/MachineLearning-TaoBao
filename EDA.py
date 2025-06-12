
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta
from ydata_profiling import ProfileReport


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('ggplot')


def load_and_preprocess(filepath):

    df = pd.read_csv(filepath, parse_dates=['datetime'])

    # 数据清洗
    df = df.sort_values(['user_id', 'datetime'])
    df = df.drop_duplicates(['user_id', 'item_id', 'behavior_type', 'datetime'])

    # 标记下一行为
    df['next_behavior'] = df.groupby('user_id')['behavior_type'].shift(-1)
    df['next_datetime'] = df.groupby('user_id')['datetime'].shift(-1)

    return df


def calculate_valid_conversions(df, conversion_window=7):

    # 标记有效转化（收藏后指定时间内购买）
    df['valid_fav_to_buy'] = (
            (df['behavior_type'] == 'fav') &
            (df['next_behavior'] == 'buy') &
            ((df['next_datetime'] - df['datetime']) <= timedelta(days=conversion_window))
    )

    # 标记有效转化（加购后指定时间内购买）
    df['valid_cart_to_buy'] = (
            (df['behavior_type'] == 'cart') &
            (df['next_behavior'] == 'buy') &
            ((df['next_datetime'] - df['datetime']) <= timedelta(days=conversion_window))
    )

    return df


def plot_behavior_distribution(df):

    # 统计各行为类型次数
    behavior_counts = df['behavior_type'].value_counts()

    # 行为类型中文映射
    behavior_names = {
        'pv': '浏览',
        'cart': '加购',
        'fav': '收藏',
        'buy': '购买'
    }
    behavior_counts.index = behavior_counts.index.map(behavior_names)

    # 创建图表
    plt.figure(figsize=(10, 6))
    bars = plt.bar(behavior_counts.index, behavior_counts.values,
                   color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])

    # 添加数据标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:,}',
                 ha='center', va='bottom', fontsize=12)

    # 图表装饰
    plt.title('用户行为类型分布', fontsize=16, pad=20)
    plt.xlabel('行为类型', fontsize=12)
    plt.ylabel('发生次数', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 保存图表
    plt.savefig('behavior_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()


def behavioral_analysis(df):

    # 基础行为统计
    behavior_stats = df['behavior_type'].value_counts().to_frame('行为次数')
    behavior_stats['行为占比'] = behavior_stats['行为次数'] / len(df) * 100

    # 有效转化统计
    metrics = {
        '点击次数': df[df['behavior_type'] == 'pv'].shape[0],
        '加购次数': df[df['behavior_type'] == 'cart'].shape[0],
        '收藏次数': df[df['behavior_type'] == 'fav'].shape[0],
        '购买次数': df[df['behavior_type'] == 'buy'].shape[0],
        '有效加购→购买次数': df['valid_cart_to_buy'].sum(),
        '有效收藏→购买次数': df['valid_fav_to_buy'].sum()
    }

    # 计算转化率（基于行为次数）
    metrics['点击→加购转化率'] = metrics['加购次数'] / metrics['点击次数'] * 100
    metrics['加购→收藏转化率'] = metrics['收藏次数'] / metrics['加购次数'] * 100
    metrics['加购→购买转化率'] = metrics['有效加购→购买次数'] / metrics['加购次数'] * 100
    metrics['收藏→购买转化率'] = metrics['有效收藏→购买次数'] / metrics['收藏次数'] * 100

    return behavior_stats, metrics


def plot_corrected_funnel(metrics):
    """绘制修正后的转化漏斗图"""
    plt.figure(figsize=(12, 6))

    # 总行为次数漏斗
    steps = ['点击', '加购', '收藏', '购买']
    values = [metrics['点击次数'], metrics['加购次数'],
              metrics['收藏次数'], metrics['购买次数']]
    plt.plot(steps, values, 'bo-', markersize=8, label='总行为次数')

    # 有效转化次数漏斗
    valid_values = [metrics['点击次数'], metrics['加购次数'],
                    metrics['收藏次数'], metrics['有效收藏→购买次数']]
    plt.plot(steps, valid_values, 'ro--', markersize=8, label='有效转化次数')

    # 添加转化率标签
    for i in range(len(steps) - 1):
        total_rate = values[i + 1] / values[i] * 100
        valid_rate = valid_values[i + 1] / valid_values[i] * 100 if i == 2 else np.nan

        plt.text(i + 0.1, values[i] * 0.9, f'总转化:{total_rate:.1f}%',
                 ha='left', color='blue')
        if i == 2:  # 仅显示收藏→购买的有效转化率
            plt.text(i + 0.1, values[i] * 0.7, f'有效转化:{valid_rate:.1f}%',
                     ha='left', color='red')

    plt.title('用户行为转化漏斗（区分总转化与有效转化）', fontsize=14, pad=20)
    plt.ylabel('行为次数', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('corrected_funnel.png', dpi=300, bbox_inches='tight')
    plt.show()


def temporal_analysis(df):
    """时间维度分析"""
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek

    # 分时段分析
    hourly_activity = df.groupby('hour')['user_id'].nunique()

    plt.figure(figsize=(12, 5))
    hourly_activity.plot(kind='bar', color='#2ca02c')
    plt.title('分时段活跃用户数', fontsize=14, pad=20)
    plt.xlabel('小时', fontsize=12)
    plt.ylabel('独立用户数', fontsize=12)
    plt.xticks(rotation=0)
    plt.savefig('hourly_activity.png', dpi=300)
    plt.show()

    return hourly_activity


def save_metrics(metrics, filename='key_metrics.txt'):
    """保存关键指标"""
    with open(filename, 'w') as f:
        f.write("关键业务指标\n")
        f.write("=" * 50 + "\n")
        for k, v in metrics.items():
            if '率' in k:
                f.write(f"{k}: {v:.2f}%\n")
            else:
                f.write(f"{k}: {v:,}\n")


def main():
    # 1. 数据加载与预处理
    df = load_and_preprocess('cleaned_UserBehavior_1M.csv')

    # 2. 计算有效转化
    df = calculate_valid_conversions(df, conversion_window=7)

    # 3. 绘制行为分布图
    plot_behavior_distribution(df)

    # 4. 行为分析
    behavior_stats, metrics = behavioral_analysis(df)

    print("=" * 50)
    print("行为类型分布:")
    print(behavior_stats)

    print("\n转化指标:")
    for k, v in metrics.items():
        print(f"{k}: {v:,.2f}{'%' if '率' in k else ''}")

    # 5. 可视化分析
    plot_corrected_funnel(metrics)
    temporal_analysis(df)

    # 6. 保存结果
    save_metrics(metrics)
    print("\n分析完成！关键指标已保存到 key_metrics.txt")

    # 7. 生成详细报告
    print("\n生成详细分析报告中...")
    profile = ProfileReport(df[['user_id', 'item_id', 'behavior_type', 'datetime']],
                            title='淘宝用户行为分析报告',
                            minimal=True)
    profile.to_file('user_behavior_report.html')


if __name__ == "__main__":
    main()