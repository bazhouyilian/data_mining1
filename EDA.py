import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.font_manager import FontProperties
import json

#设置全局字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号

sns.set(style="whitegrid")
font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf") 
path = 'data/1G_data/part-00007.parquet'
df = pd.read_parquet(path) 

# 查看前几行
print(df.head())

# # 查看性别具体的值
print(df['purchase_history'].value_counts())

# # # 查看登记时间的类型
# print(df['registration_date'].value_counts())

# print(df['is_active'].value_counts())
# 查看基本信息
print(df.info())

def extract_purchase_info(purchase):
    try:
        record = json.loads(purchase)
        avg_price = record.get('average_price', 0)
        item_count = len(record.get('items', []))
        category = record.get('category', '未知')
        return pd.Series([avg_price, item_count, category])
    except:
        print(f"解析失败: {purchase}, 错误信息: {e}")
        return pd.Series([0, 0, '未知'])

df[['purchase_avg_price', 'purchase_item_count', 'purchase_category']] = df['purchase_history'].apply(extract_purchase_info)
dig_data = ['age', 'income', 'credit_score', 'purchase_avg_price', 'purchase_item_count']

print(df[dig_data].describe())


#绘制基本属性相关的图
def draw_basic_attributes_distribution(df):
        
    # 创建子图：2 行 3 列
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("用户基础属性分布", fontsize=16,fontproperties=font)

    # 1. 年龄分布
    sns.histplot(df['age'], bins=30, kde=True, ax=axes[0, 0], color='skyblue')
    axes[0, 0].set_title("年龄分布", fontproperties=font)

    # 2. 收入分布
    sns.histplot(df['income'], bins=30, kde=True, ax=axes[0, 1], color='salmon')
    axes[0, 1].set_title("收入分布",fontproperties=font)

    # 3. 性别分布
    sns.countplot(x='gender', data=df, ax=axes[0, 2], palette='Set2')
    axes[0, 2].set_title("性别分布", fontproperties=font )
    axes[0, 2].set_xticklabels(axes[0, 2].get_xticklabels(), fontproperties=font)

    # 4. 国家分布（取前10国家）
    top_countries = df['country'].value_counts().nlargest(10)
    sns.barplot(x=top_countries.values, y=top_countries.index, ax=axes[1, 0], palette='viridis')
    axes[1, 0].set_title("国家分布（前十）", fontproperties=font)
    axes[1, 0].set_yticklabels(axes[1, 0].get_yticklabels(), fontproperties=font)

    # 5. 信用分分布
    sns.histplot(df['credit_score'], bins=30, kde=True, ax=axes[1, 1], color='mediumseagreen')
    axes[1, 1].set_title("信用分分布", fontproperties=font)

    # 第六个子图隐藏
    axes[1, 2].axis('off')

    # 自动调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# draw_basic_attributes_distribution(df)
def draw_user_behavior_distribution(df):
    # 提取用户行为字段


    # 创建图形
    fig, axes = plt.subplots(2, 3, figsize=(16, 12))
    fig.suptitle("用户行为可视化", fontsize=16, fontproperties=font)

    # 1. 平均购买价格 - 箱线图
    sns.boxplot(y='purchase_avg_price', data=df, ax=axes[0, 0], color='lightblue')
    axes[0, 0].set_title("平均购买价格分布（箱线图）", fontproperties=font)

    # 2. 购买商品数量 - 直方图
    sns.histplot(df['purchase_item_count'], bins=20, kde=False, ax=axes[0, 1], color='salmon')
    axes[0, 1].set_title("每个用户的购买数量分布", fontproperties=font)

    # 3. 主要消费品类 - 柱状图
    top_categories = df['purchase_category'].value_counts().nlargest(10)
    sns.barplot(x=top_categories.index, y=top_categories.values, ax=axes[1, 0], palette='Set3')
    axes[1, 0].set_title("用户主要消费品类分布", fontproperties=font)
    axes[1, 0].tick_params(axis='x', rotation=30)
    axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), fontproperties=font)

    # 4. 收入 vs 信用分数 - 散点图
    sns.scatterplot(x='income', y='credit_score', data=df, alpha=0.4, ax=axes[1, 1])
    axes[1, 1].set_title("财务状况（收入 vs 信用分）", fontproperties=font)
    correlation = df[['income', 'credit_score']].corr(method='spearman').iloc[0, 1]
    print(f"收入与信用分数的斯皮尔曼相关系数为: {correlation:.2f}")

    # 5. 收入直方图
    sns.histplot(df['income'], bins=30, kde=False, ax=axes[1, 2], color='salmon')
    axes[1, 2].set_title("收入分布", fontproperties=font)


    axes[0, 2].axis('off')

    # 自动调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    return df

# df = draw_user_behavior_distribution(df)
# #绘制相关性热力图
def draw_correlation_heatmap(df):
    selected_cols = ['income', 'credit_score', 'purchase_avg_price', 'purchase_item_count']
    df_selected = df[selected_cols].dropna()  # 去掉缺失值行

 # 计算斯皮尔曼相关性矩阵
    corr_matrix = df_selected.corr(method='spearman')

    # 绘制热力图
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title("用户关键指标相关性热力图", fontproperties=font)
    plt.tight_layout()
    plt.show()

# draw_correlation_heatmap(df)