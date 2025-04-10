import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import glob
import os
import re
from matplotlib.font_manager import FontProperties
import json
from sklearn.preprocessing import MinMaxScaler
import time
#设置全局字体
plt.rcParams["font.family"] = "Arial"  #仿宋
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号
font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf")  # SimHei 字体路径

sns.set(style="whitegrid")
def handle_missing_values(df_clean, file_id):
    missing = df_clean.isnull().sum()
    missing_ratio = missing / len(df_clean) * 100
    missing_info = pd.DataFrame({
        '字段': missing.index,
        '缺失值数量': missing.values,
        '缺失值比例 (%)': missing_ratio.values,
        '文件': file_id
    })
    # 打印缺失值信息表格
    print("缺失值统计：")
    print(missing_info)
    df_clean = df_clean.dropna()
    print(f"处理空缺值后剩余记录数: {len(df_clean)}\n")
    return df_clean, missing_info

# #缺失值处理
# def handle_missing_values(df_clean):
#     missing = df_clean.isnull().sum()
#     # 计算缺失值比例
#     missing_ratio = missing / len(df_clean) * 100
    
#     # 创建一个 DataFrame 来存储缺失值信息
#     missing_info = pd.DataFrame({
#         '缺失值数量': missing,
#         '缺失值比例 (%)': missing_ratio
#     })
    
#     # 打印缺失值信息表格
#     print("缺失值统计：")
#     print(missing_info)
    
#     # 处理空缺值
#     df_clean = df_clean.dropna()
#     print(f"处理空缺值后剩余记录数: {len(df_clean)}\n")
#     return df_clean
def normalize_numeric_columns(df, columns):
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

def extract_purchase_features(df):
    """从 purchase_history 中提取 average_price 和 item_count"""
    avg_prices = []
    item_counts = []

    for record in df['purchase_history']:
        try:
            data = json.loads(record)
            avg_prices.append(data.get('average_price', np.nan))
            item_counts.append(len(data.get('items', [])))
        except Exception:
            avg_prices.append(np.nan)
            item_counts.append(np.nan)

    df['purchase_avg_price'] = avg_prices
    df['purchase_item_count'] = item_counts
    return df
def is_valid_phone(phone):
    return bool(re.match(r"\d{3}-\d{3}-\d{4}$", phone))
def is_valid_email(email):
    return bool(re.match(r"[^@]+@[^@]+\.[^@]+", email))


def remove_outliers_iqr(df, column):
    """使用 IQR 方法去除某列的异常值"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = max(Q1 - 1.5 * IQR, 0)
    upper_bound = Q3 + 1.5 * IQR
    print(f"{column} 过滤范围: {lower_bound:.2f} ~ {upper_bound:.2f}")
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def draw_boxplot(df, col, file_id, info):
    plt.figure(figsize=(6, 4))
    sns.boxplot(y=df[col])
    plt.title(f"{col} 箱线图 - {file_id}", fontproperties=font)
    plt.tight_layout()

    # 构建保存路径
    save_path = f"./result_images/{info}boxplot_{col}_{file_id}.png"
    save_dir = os.path.dirname(save_path)
    
    # 如果目录不存在，则创建目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    plt.savefig(save_path)  # 保存为图片
    plt.close()


# 文件夹路径
info = '30G_data'
folder_path = 'data/' + info

# 新目录路径
cleaned_folder = 'data/cleaned/' + info
os.makedirs(cleaned_folder, exist_ok=True)  # 自动创建目录（如果不存在）

# 获取所有 parquet 文件路径
parquet_files = sorted(glob.glob(os.path.join(folder_path, 'part-*.parquet')))

all_high_value_users = []
missing_summary_list = []
outlier_summary_list = []

#记录开始时间并打印
start_time = time.time()
print("开始时间:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
for file in parquet_files:
    print("*****************************************************************")
    print(f"正在处理文件: {file}")
    file_id = os.path.basename(file).split('.')[0]  # 如 part-00000
    filename = os.path.basename(file)
    try:
        # 读取 parquet 文件
        df_clean = pd.read_parquet(file, engine='pyarrow')
        print(df_clean.info())

        df_clean, missing_info = handle_missing_values(df_clean, file_id)
        missing_summary_list.append(missing_info)

    # 处理数值异常值字段
        for col in ['age', 'income', 'credit_score']:
            print(f"\n处理字段：{col}")
            before = len(df_clean)
            df_clean = remove_outliers_iqr(df_clean, col)
            after = len(df_clean)
            removed = before - after
            print(f"删除异常值后：{before - after} 行被移除")
            ratio = round(removed / before * 100, 2)
            #计算异常值比例
            print(f"异常值比例：{round((after - before) / before * 100, 2)}%")
            draw_boxplot(df_clean, col, file_id, info)

            outlier_summary_list.append({
            '文件': file_id,
            '字段': col,
            '异常值数量': removed,
            '异常值比例 (%)': ratio
            })
    # 处理非数值字段
      
        expected_cols = ['gender', 'email', 'phone_number']
        existing = set(df_clean.columns).intersection(expected_cols)

        if 'gender' in existing:
            t1 = len(df_clean)
            df_clean = df_clean[df_clean['gender'].isin(['男', '女'])]
            t2 = len(df_clean)
            print(f"删除性别异常值后：{t1 - t2} 行被移除")

            outlier_summary_list.append({
            '文件': file_id,
            '字段': 'gender',
            '异常值数量': t1 - t2,
            '异常值比例 (%)': (t1 - t2) / t1 * 100
            })

        if 'email' in existing:
            t1  = len(df_clean)
            df_clean = df_clean[df_clean['email'].apply(is_valid_email)]
            t2 = len(df_clean)
            print(f"删除邮箱异常值后：{t1 - t2} 行被移除")
            outlier_summary_list.append({
            '文件': file_id,
            '字段': 'email',
            '异常值数量': t1 - t2,
            '异常值比例 (%)': (t1 - t2) / t1 * 100
            })
        if 'phone_number' in existing:
            t1 = len(df_clean)
            df_clean = df_clean[df_clean['phone_number'].apply(is_valid_phone)]
            t2 = len(df_clean)
            print(f"删除电话号码异常值后：{t1 - t2} 行被移除")
            outlier_summary_list.append({
            '文件': file_id,
            '字段': 'phone_number',
            '异常值数量': t1 - t2,
            '异常值比例 (%)': (t1 - t2) / t1 * 100
            })
         # 提取购买信息
        df_clean = extract_purchase_features(df_clean)

        #处理登记时间
        # 确保 registration_date 是 datetime 类型
        df_clean['registration_date'] = pd.to_datetime(df_clean['registration_date'])
        # 获取当前日期
        now = pd.to_datetime("today")
        # 计算注册天数（注册到现在经过了多少天）
        df_clean['registration_days'] = (now - df_clean['registration_date']).dt.days


        # 选择要归一化的字段
        numeric_columns = ['registration_days', 'age', 'income', 'credit_score', 'purchase_avg_price', 'purchase_item_count']
         # 归一化数值字段
        df_clean = normalize_numeric_columns(df_clean, numeric_columns)
        print(f"✅ 数据已归一化")

        df_clean['is_active_num'] = df_clean['is_active'].astype(int)

        #计算综合得分
        df_clean['user_value_score'] = (
        0.25 * df_clean['income'] +
        0.20 * df_clean['credit_score'] +
        0.20 * df_clean['purchase_avg_price'] +
        0.15 * df_clean['purchase_item_count'] +
        0.10 * df_clean['is_active_num'] +
        0.10 * (1 - df_clean['registration_days'])  # 注册越早分数越高
        )

        threshold = df_clean['user_value_score'].quantile(0.90)
        high_value_users = df_clean[df_clean['user_value_score'] >= threshold]
        print(f"🎯 高价值用户数量：{len(high_value_users)}")

        # 加入列表
        all_high_value_users.append(high_value_users)
    except Exception as e:
        print(f"读取失败: {file}，错误：{e}")


# 拼接所有高价值用户
final_high_value_users = pd.concat(all_high_value_users, ignore_index=True)

# 保存到文件
output_path = 'data/' + info + 'high_value_users.parquet'
final_high_value_users.to_parquet(output_path, engine='pyarrow', index=False)
print(f"✅ 所有高价值用户数据已保存到：{output_path}")

# 缺失值总表
missing_summary_df = pd.concat(missing_summary_list, ignore_index=True)
missing_summary_df = missing_summary_df[missing_summary_df['缺失值数量'] > 0]
missing_summary_df.to_csv('data/' + info + 'result_missing_summary.csv', index=False, encoding='utf-8-sig')
print("📊 缺失值统计表已保存为 result_missing_summary.csv")

# 异常值总表
outlier_summary_df = pd.DataFrame(outlier_summary_list)
outlier_summary_df = outlier_summary_df[outlier_summary_df['异常值数量'] > 0]
outlier_summary_df.to_csv('data/' + info + 'result_outlier_summary.csv', index=False, encoding='utf-8-sig')
print("📊 异常值统计表已保存为 result_outlier_summary.csv")

#记录结束时间并打印
end_time = time.time()
print("结束时间:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
#打印总时长
total_time = end_time - start_time
#打印总时长，以分钟为单位
print("总耗时:", total_time / 60, "分钟")
