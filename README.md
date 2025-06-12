# MachineLearning-TaoBao
# 淘宝用户行为分析项目

## 项目结构
- data-read.py: 数据清洗脚本
- EDA.py: 探索性数据分析脚本
- feature-first.py: 特征工程实现
- model-first.py: 建模与评估代码
- New-feature.py: 优化版特征工程
- New-model.py: 优化版建模代码
- New-model.py: 优化版建模代码

## 数据文件
- UserBehavior.csv: 原始数据
- cleaned_UserBehavior_1M.csv: 清洗后数据
- featured_data.csv: 特征工程结果
- high_value_conversion_features_final.csv：优化版特征工程结果


## 运行步骤
1. 运行Data-read.py进行数据清洗
2. 运行EDA.py生成分析报告
3. 执行Feature-first.py和model-first.py进行基础建模
4. 运行New-feature.py和New-model.py获取优化版结果

## 依赖环境
Python , Pandas , Numpy , Scikit-learn , XGBoost 

## 结果查看
- behavior_distribution.png: 用户行为分布图
- corrected_funnel.png: 转化漏斗图
- hourly_activity.png:各时段用户活跃度
- user_behavior_report.html:Eda可视化报告
- XGBoost_performance.png：第一版结果
- 逻辑回归_performance.png：第一版结果
- 随机森林_performance.png：第一版结果
- model_results.txt：优化版模型结果
