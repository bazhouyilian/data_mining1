
import os
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import json
import time
import tqdm
from matplotlib.font_manager import FontProperties

folder_path = 'data/30G_data'
font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf") 

# 初始化字典：用 Counter 自动计数
category_stats = {
    'country': Counter(),
    'gender': Counter()
}

# 定义需要统计的字段和各自的分箱配置
bin_config = {
    'age': {
        'bins': list(range(18, 101, 10)) + [np.inf],  # 18, 28, ..., 98, 101 -> 共10个边界 = 9个区间
        'labels': [f"{i}-{i+9}" for i in range(18, 100, 10)]  # 加上最后一个标签
    },
    'income': {
        'bins': list(range(0, 1000000, 100000)) + [np.inf],  # 0, 20000, ..., 200000, inf -> 共11个边界 = 10个区间
        'labels': [f"{i}-{i+99999}" for i in range(0, 1000000, 100000)] # 加上最后一个标签
    },
    'purchase_avg_price': {
        'bins': list(range(0, 10001, 1000)) + [np.inf],
        'labels': [f"{i}-{i+999}" for i in range(0, 10000, 1000)] + ["10000+"]
    },
    'session_duration': {
        'bins': list(range(0, 121, 10)) + [np.inf],
        'labels': [f"{i}-{i+9}" for i in range(0, 120, 10)] + ["120+"]
    },
    'login_count': {
        'bins': list(range(0, 101, 10)) + [np.inf],
        'labels': [f"{i}-{i+9}" for i in range(0, 100, 10)] + ["100+"]
    }
}

print(bin_config)
# 初始化总统计结果
total_stats = {field: defaultdict(int) for field in bin_config}



#开始时间
start_time = time.time()
print("开始时间:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

sampled_points = [] 

# count = 0
# 遍历所有 parquet 文件
for file in os.listdir(folder_path):
    if file.endswith('.parquet'):
        file_path = os.path.join(folder_path, file)
        print(f"📂 处理文件: {file_path}")


        try:
            df = pd.read_parquet(file_path, engine='pyarrow')
            # 用于存放解析结果
            purchase_avg_price = []
            item_count = []
            category_list = []
            country_counter = Counter()
            gender_counter = Counter()
            session_duration_list = []
            login_count_list = []

            # 遍历 purchase_history，避免 df.loc[i] 逐行慢操作
            for i, row in tqdm.tqdm(df.iterrows(), total=len(df)):
                try:
                    purchase = json.loads(row['purchase_history'])
                    avg_price = purchase.get('avg_price', 0)
                    items = purchase.get('items', [])
                    category = purchase.get('categories', '未知')

              
                    # 提取 login_history 信息
                    login = json.loads(row['login_history'])
                    session_duration = login.get('avg_session_duration', np.nan)
                    login_count = login.get('login_count', np.nan)

                    session_duration_list.append(session_duration)
                    login_count_list.append(login_count)

                    purchase_avg_price.append(avg_price)
                    item_count.append(len(items))
                    category_list.append(category)

                    # 统计类别频次
                    country_counter[row['country']] += 1
                    gender_counter[row['gender']] += 1

                except Exception as e:
                    purchase_avg_price.append(np.nan)
                    item_count.append(np.nan)
                    category_list.append('解析失败')
                    print(f"解析失败: {file_path}, 错误: {e}")

            # 一次性赋值（比逐行 df.loc 快很多）
            df['purchase_avg_price'] = purchase_avg_price
            df['item_count'] = item_count
            df['category'] = category_list
            df['session_duration'] = session_duration_list
            df['login_count'] = login_count_list
            print(df['purchase_avg_price'].value_counts())

            # 合并频次到全局统计
            for k, v in country_counter.items():
                category_stats['country'][k] += v
            for k, v in gender_counter.items():
                category_stats['gender'][k] += v

            df_sample = df[['income', 'purchase_avg_price']].dropna()
            df_sample = df_sample.sample(n=10000, random_state=42)
            print(df_sample['purchase_avg_price'].value_counts())
            print(df_sample['income'].value_counts())
            sampled_points.append(df_sample)

            for field, cfg in bin_config.items():
                if field not in df.columns:
                    print(f"⚠️ 字段 {field} 不存在，跳过")
                    continue

                # 分箱统计
                binned = pd.cut(df[field], bins=cfg['bins'], labels=cfg['labels'], right=False)
                value_counts = binned.value_counts().to_dict()

                # 累加统计结果
                for label in cfg['labels']:
                    total_stats[field][label] += value_counts.get(label, 0)

        except Exception as e:
            print(f"❌ 读取失败: {file_path}, 错误: {e}")
            continue

        
        # count += 1
        # if count == 1:
        #     break

# 🔍 打印汇总统计结果
for field in bin_config:
    print(f"\n📊 字段【{field}】分布统计：")
    for label in bin_config[field]['labels']:
        print(f"  {label}: {total_stats[field][label]}")

# 计算图像布局行列数（例如最多每行2个图）
num_fields = len(bin_config)
cols = 2
rows = (num_fields + cols - 1) // cols  # 自动计算需要几行

fig, axs = plt.subplots(rows, cols, figsize=(12, 4 * rows))  # 每行高度4
axs = axs.flatten()  # 展平成一维，方便索引

# 遍历所有数值字段，分别画图
for idx, field in enumerate(bin_config.keys()):
    labels = bin_config[field]['labels']
    counts = [total_stats[field][label] for label in labels]

    axs[idx].bar(labels, counts)
    axs[idx].set_title(f"{field.capitalize()} Distribution")
    axs[idx].set_xlabel("Range")
    axs[idx].set_ylabel("Count")
    axs[idx].tick_params(axis='x', rotation=45)

# 清除多余的子图框
for i in range(num_fields, len(axs)):
    fig.delaxes(axs[i])

plt.tight_layout()
# 保存图像
plt.savefig("./result_images/30G直方图.png", dpi=300, bbox_inches='tight')
plt.close()

# 取出频数字典
top_countries = category_stats['country'].most_common(10)
genders = category_stats['gender'].most_common()

# 拆分为标签和数量
country_labels, country_counts = zip(*top_countries)
gender_labels, gender_counts = zip(*genders)

# 创建子图
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# 国家分布
axs[0].bar(country_labels, country_counts)
axs[0].set_title("Top 10 Countries")
axs[0].set_xlabel("Country", fontproperties=font)
axs[0].set_ylabel("Count")
axs[0].tick_params(axis='x', rotation=45)
axs[0].set_xticklabels(axs[0].get_xticklabels(), fontproperties=font)

# 性别分布
axs[1].bar(gender_labels, gender_counts, color='orange')
axs[1].set_title("Gender Distribution")
axs[1].set_xlabel("Gender", fontproperties=font)
axs[1].set_ylabel("Count")
axs[1].set_xticklabels(axs[1].get_xticklabels(), fontproperties=font)

plt.tight_layout()
# 保存图像
plt.savefig("./result_images/30G柱形图.png", dpi=300, bbox_inches='tight')
plt.close()

combined_df = pd.concat(sampled_points, ignore_index=True)
print(f"✅ 总采样点数量: {len(combined_df)}")
print(combined_df['income'].value_counts())
print(combined_df['purchase_avg_price'].value_counts())

plt.figure(figsize=(10, 6))
plt.scatter(
    combined_df['income'],
    combined_df['purchase_avg_price'],
    alpha=0.3,
    s=10,
    c='steelblue'
)
plt.title("Income vs Purchase Avg Price")
plt.xlabel("Income")
plt.ylabel("Purchase Avg Price")
plt.grid(True)
plt.tight_layout()
# 保存图像
plt.savefig("./result_images/30G散点图.png", dpi=300, bbox_inches='tight')
plt.close()

#记录结束时间并打印
end_time = time.time()
print("结束时间:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
#打印总时长
total_time = end_time - start_time
#打印总时长，以分钟为单位
print("总耗时:", total_time / 60, "分钟")