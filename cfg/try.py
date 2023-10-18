import random
from collections import defaultdict

# 原始数据字典
data = {
    'bathtub': ['004239', '002176', '002134', ...],  # 你的数据
    'bed': ['000745', '000842', '002603', ...],  # 你的数据
    'bookshelf': ['012345', '012346', '012347', ...],  # 你的数据
    # 其他键值对省略
}

# 函数：将数据转换成新字典的格式
def transform_data(data):
    new_data = {}
    for key, values in data.items():
        for value in values:
            if value in new_data:
                new_data[value][key] = 1
            else:
                new_data[value] = {k: 0 for k in data.keys()}
                new_data[value][key] = 1
    return new_data

# 函数：随机选择5%的数据
def select_random_samples(data, percent_to_select):
    all_samples = list(data.keys())
    selected_samples = random.sample(all_samples, int(len(all_samples) * percent_to_select))
    return {sample: data[sample] for sample in selected_samples}

# 函数：计算每个键对应的值的总和
def calculate_totals(selected_data):
    totals = defaultdict(int)
    for sample_data in selected_data.values():
        for key, value in sample_data.items():
            totals[key] += value
    return totals

# 初始化选择的数据为空
selected_data = {}

# 持续选择数据，直到满足条件
while True:
    # 步骤1：将数据转换成新字典的格式
    new_data = transform_data(data)
    
    # 清空选择的数据
    selected_data = {}
    
    # 步骤2：随机选择5%的数据
    selected_data = select_random_samples(new_data, 0.5)
    
    # 步骤3：计算每个键对应的值的总和
    totals = calculate_totals(selected_data)
    
    # 步骤4：检查是否满足条件（每个键对应的值都大于0）
    if all(value > 0 for value in totals.values()):
        break

# 打印每个键对应的值的总和
print(selected_data)
# 初始化一个空的新字典，用于存储转换后的数据
converted_data = {key: [] for key in data.keys()}

# 遍历最终生成的字典
for sample, values in selected_data.items():
    # 遍历每个键
    for key, value in values.items():
        # 如果值为1，将该样本添加到相应的键中
        if value == 1:
            converted_data[key].append(sample)

# 打印转换后的数据
print(converted_data)
