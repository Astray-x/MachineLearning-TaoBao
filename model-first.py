import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report,
                             precision_recall_curve, average_precision_score,
                             confusion_matrix, roc_auc_score, f1_score, roc_curve)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns

# 1. 样式设置
try:
    plt.style.use('seaborn-v0_8')  # 新版本Matplotlib
except:
    plt.style.use('ggplot')  # 备用样式

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 2. 数据准备
def load_data():
    """加载并检查数据"""
    if not os.path.exists('featured_data.csv'):
        raise FileNotFoundError("未找到数据文件 featured_data.csv，请确保文件路径正确")

    # 尝试多种编码方式读取文件
    encodings = ['utf-8', 'gbk', 'latin1']
    df = None
    for encoding in encodings:
        try:
            df = pd.read_csv('featured_data.csv', encoding=encoding)
            break
        except UnicodeDecodeError:
            continue

    if df is None:
        raise ValueError("无法读取文件，请检查文件编码")

    print("\n数据加载成功，前5行样例:")
    print(df.head())

    X = df.drop(['user_id', 'buy_count'], axis=1)
    y = (df['buy_count'] > 0).astype(int)

    # 检查数据平衡性
    print(f"\n类别分布: 正样本 {y.mean():.2%}, 负样本 {1 - y.mean():.2%}")
    return X, y


# 3. 可视化报告生成
def generate_model_report(model_name, y_test, y_pred, y_prob, cv_scores):
    """生成模型评估的可视化报告"""
    fig = plt.figure(figsize=(15, 10))

    # 创建子图布局
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)  # PR曲线
    ax2 = plt.subplot2grid((2, 2), (1, 0))  # ROC曲线
    ax3 = plt.subplot2grid((2, 2), (1, 1))  # 混淆矩阵

    # 绘制PR曲线
    precisions, recalls, _ = precision_recall_curve(y_test, y_prob)
    ap_score = average_precision_score(y_test, y_prob)
    ax1.step(recalls, precisions, where='post', label=f'{model_name} (AP={ap_score:.2f})')
    ax1.set_xlabel('召回率')
    ax1.set_ylabel('精确率')
    ax1.set_title('精确率-召回率曲线')
    ax1.legend(loc='lower left')
    ax1.grid(True)

    # 绘制ROC曲线（修复roc_curve未定义错误）
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    ax2.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    ax2.plot([0, 1], [0, 1], 'k--')
    ax2.set_xlabel('假正率')
    ax2.set_ylabel('真正率')
    ax2.set_title('ROC曲线')
    ax2.legend(loc='lower right')
    ax2.grid(True)

    # 绘制混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3, cbar=False)
    ax3.set_title('混淆矩阵')
    ax3.set_xticklabels(['预测负类', '预测正类'])
    ax3.set_yticklabels(['真实负类', '真实正类'])

    # 添加交叉验证结果
    fig.text(0.15, 0.05, f'交叉验证AUC: {np.mean(cv_scores):.3f} (±{np.std(cv_scores):.3f})',
             fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f'{model_name}_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已生成可视化报告: {model_name}_performance.png")


# 4. 模型训练与评估
def train_and_evaluate(X, y):
    """完整的模型训练流程"""
    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    # 检查数据平衡性
    print(f"\n训练集类别分布: 正样本 {y_train.mean():.2%}, 负样本 {1 - y_train.mean():.2%}")

    # 处理类别不平衡
    print("\n应用SMOTE过采样...")
    try:
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        print(f"过采样后训练集形状: {X_res.shape}")
    except Exception as e:
        print(f"SMOTE过采样失败: {str(e)}")
        X_res, y_res = X_train, y_train

    # 定义模型
    models = {
        "逻辑回归": LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        ),
        "随机森林": RandomForestClassifier(
            class_weight='balanced_subsample',
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
            eval_metric='logloss',
            n_estimators=200,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1
        )
    }

    # 训练评估每个模型
    for name, model in models.items():
        print(f"\n{'=' * 30}\n正在训练模型: {name}")

        try:
            # 交叉验证
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X_res, y_res, cv=cv, scoring='roc_auc')
            print(f"交叉验证AUC: {np.mean(cv_scores):.3f} (±{np.std(cv_scores):.3f})")

            # 训练模型
            model.fit(X_res, y_res)

            # 预测概率
            y_prob = model.predict_proba(X_test)[:, 1]

            # 自动选择最佳阈值
            precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
            best_threshold = thresholds[np.argmax(f1_scores)]
            y_pred = (y_prob >= best_threshold).astype(int)

            # 打印评估结果
            print(f"\n最佳阈值: {best_threshold:.4f}")
            print(classification_report(y_test, y_pred, zero_division=0))

            # 生成可视化报告
            generate_model_report(name, y_test, y_pred, y_prob, cv_scores)

        except Exception as e:
            print(f"模型 {name} 训练失败: {str(e)}")

    print("\n所有模型评估完成，请查看生成的报告图片")


# 主程序
if __name__ == "__main__":
    try:
        print("=" * 50)
        print("电商用户行为分析模型训练开始")
        print("=" * 50)

        X, y = load_data()
        train_and_evaluate(X, y)

    except Exception as e:
        print(f"\n发生错误: {str(e)}")
    finally:
        input("\n按Enter键退出程序...")