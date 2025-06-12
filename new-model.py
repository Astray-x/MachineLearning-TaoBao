import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os
from datetime import datetime

CONFIG = {
    'feature_file': 'high_value_conversion_features_final.csv',
    'model_file': 'conversion_model.pkl',
    'result_file': 'model_results.txt',
    'test_size': 0.2,
    'random_state': 42,
    'target': 'next_is_conversion'  # 可改为next_high_value
}


def init_output():

    with open(CONFIG['result_file'], 'w') as f:
        f.write(f"建模结果报告 {datetime.now()}\n")
        f.write("=" * 50 + "\n")
        f.write(f"特征文件: {CONFIG['feature_file']}\n")
        f.write(f"目标变量: {CONFIG['target']}\n\n")


def log_to_file(content):

    with open(CONFIG['result_file'], 'a') as f:
        f.write(content + "\n")
    print(content)


def load_features():

    try:
        df = pd.read_csv(CONFIG['feature_file'])
        log_to_file(f"数据维度: {df.shape}")
        log_to_file(f"特征列表: {df.columns.tolist()}")
        return df
    except Exception as e:
        log_to_file(f"加载失败: {str(e)}")
        raise


def prepare_data(df):

    # 选择特征列（根据实际特征调整）
    features = [
        'minute_sin', 'minute_cos', 'is_peak_hour',
        'pv_count', 'fav_count', 'cart_count',
        'pv_to_cart_rate', '24h_high_ratio',
        'pv_to_cart_speed'
    ]

    # 检查缺失特征
    missing = [f for f in features if f not in df.columns]
    if missing:
        log_to_file(f"警告: 缺失特征 {missing}")
        features = [f for f in features if f not in missing]

    X = df[features]
    y = df[CONFIG['target']]

    log_to_file(f"正样本比例: {y.mean():.2%}")
    return train_test_split(
        X, y,
        test_size=CONFIG['test_size'],
        random_state=CONFIG['random_state'],
        stratify=y
    )


def train_and_evaluate(X_train, X_test, y_train, y_test):

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        max_depth=10,
        random_state=CONFIG['random_state'],
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    log_to_file("模型训练完成")

    # 评估指标
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    log_to_file("\n分类报告:")
    clf_report = classification_report(y_test, y_pred)
    log_to_file(clf_report)

    log_to_file(f"AUC得分: {roc_auc_score(y_test, y_proba):.4f}")

    # 特征重要性
    importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    log_to_file("\n特征重要性:")
    log_to_file(importance.to_string())

    return model, importance


def save_outputs(model, importance):

    # 保存模型
    joblib.dump(model, CONFIG['model_file'])
    log_to_file(f"模型已保存至 {CONFIG['model_file']}")

    # 保存重要性
    importance.to_csv('feature_importance.csv', index=False)
    log_to_file("特征重要性已保存")


def main():

    init_output()
    try:
        df = load_features()
        X_train, X_test, y_train, y_test = prepare_data(df)
        model, importance = train_and_evaluate(X_train, X_test, y_train, y_test)
        save_outputs(model, importance)
        log_to_file("\n建模流程成功完成！")
    except Exception as e:
        log_to_file(f"\n错误: {str(e)}")
        raise


if __name__ == "__main__":
    main()