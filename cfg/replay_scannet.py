import os
import numpy as np
import random
from collections import defaultdict
import pickle
from easydict import EasyDict
import random
# random.seed(10)
import csv
__C = EasyDict()
cfg = __C
percent_to_select=0.05
__C.NUM_CLASSES = 18
__C.DONOTCARE_CLASS_IDS = np.array([])
__C.NYU40IDS = np.array([36, 4, 10, 3, 5, 12, 16, 14, 8, 39, 11, 24, 28, 34, 6, 7, 33, 9]) #the corresponding NYU40 ids of interested object class
__C.TYPE_WHITELIST = ['bathtub', 'bed', 'bookshelf', 'cabinet', 'chair', 'counter', 'curtain', 'desk', 'door', 'otherfurniture',
                      'picture', 'refrigerator', 'showercurtain', 'sink', 'sofa', 'table', 'toilet', 'window']
__C.MAX_NUM_POINT = 50000

__C.TYPE_MEAN_SIZE =  {'cabinet': np.array([0.76966726, 0.81160211, 0.92573741]),
                       'bed': np.array([1.876858, 1.84255952, 1.19315654]),
                       'chair': np.array([0.61327999, 0.61486087, 0.71827014]),
                       'sofa': np.array([1.39550063, 1.51215451, 0.83443565]),
                       'table': np.array([0.97949596, 1.06751485, 0.63296875]),
                       'door': np.array([0.53166301, 0.59555772, 1.75001483]),
                       'window': np.array([0.96247056, 0.72462326, 1.14818682]),
                       'bookshelf': np.array([0.83221924, 1.04909355, 1.68756634]),
                       'picture': np.array([0.21132214, 0.4206159 , 0.53728459]),
                       'counter': np.array([1.44400728, 1.89708334, 0.26985747]),
                       'desk': np.array([1.02942616, 1.40407966, 0.87554322]),
                       'curtain': np.array([1.37664116, 0.65521793, 1.68131292]),
                       'refrigerator': np.array([0.66508189, 0.71111926, 1.29885307]),
                       'showercurtain': np.array([0.41999174, 0.37906947, 1.75139715]),
                       'toilet': np.array([0.59359559, 0.59124924, 0.73919014]),
                       'sink': np.array([0.50867595, 0.50656087, 0.30136236]),
                       'bathtub': np.array([1.15115265, 1.0546296 , 0.49706794]),
                       'otherfurniture': np.array([0.47535286, 0.49249493, 0.58021168])
                        }

__C.NUM_HEADING_BIN = 1 #Object bboxes are alix-aligned in ScanNet

__C.MAX_NUM_OBJ = 64 # maximum number of objects allowed per scene
__C.MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])

__C.NUM_BASE_CLASSES = 14
__C.BASE_TYPES = ['bathtub', 'bed', 'bookshelf', 'cabinet', 'chair', 'counter', 'curtain', 'desk', 'door','otherfurniture', 'picture', 'refrigerator', 'showercurtain', 'sink']
__C.BASE_NYUIDS = np.array([36, 4, 10, 3, 5, 12, 16, 14, 8, 39, 11, 24, 28, 34,])

__C.NUM_NOVEL_CLASSES = 4
__C.NOVEL_TYPES = [ 'bathtub', 'bed', 'bookshelf', 'cabinet', 'chair', 'counter', 'curtain', 'desk', 'door','otherfurniture', 'picture', 'refrigerator', 'showercurtain', 'sink','sofa', 'table','toilet', 'window']
__C.NOVEL_NYUIDS = np.array([ 36, 4, 10, 3, 5, 12, 16, 14, 8, 39, 11, 24, 28, 34,33, 9,6,7])

# __C.NUM_BASE_CLASSES = 9
# __C.BASE_TYPES = ['bathtub', 'bed', 'bookshelf', 'cabinet', 'chair', 'counter', 'curtain', 'desk', 'door']
# __C.BASE_NYUIDS = np.array([36, 4, 10, 3, 5, 12, 16, 14, 8])

# __C.NUM_NOVEL_CLASSES = 9
# __C.NOVEL_TYPES = ['otherfurniture', 'picture', 'refrigerator', 'showercurtain', 'sink', 'sofa', 'table', 'toilet', 'window']
# __C.NOVEL_NYUIDS = np.array([39, 11, 24, 28, 34, 6, 7, 33, 9])
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
def get_class2scans(data_path, split='train'):
    '''Generate a mapping dictionary whose key is the class name and the values are the corresponding scan names
       containing objects of this class
    '''
    index_data_path = os.path.join(data_path, 'index_data')
    class2scans_file = os.path.join(index_data_path, '%s_class2scans.pkl' %split)
    if not os.path.exists(index_data_path): os.mkdir(index_data_path)
    if os.path.exists("0.05scannet_scene"):
        print("1")
        with open("0.05scannet_scene", 'rb') as f:
            class2scans = pickle.load(f)
            for class_name in __C.TYPE_WHITELIST:
                print('\t class_name: {0} | num of scans: {1}'.format(class_name, len(class2scans[class_name])))      
            print(class2scans)     
    elif os.path.exists(class2scans_file):
        with open(class2scans_file, 'rb') as f:
            class2scans = pickle.load(f)
            for class_name in __C.TYPE_WHITELIST:
                print('\t class_name: {0} | num of scans: {1}'.format(class_name, len(class2scans[class_name])))
            # print(class2scans)
            # keys_to_process = ['bathtub', 'bed', 'bookshelf', 'cabinet', 'chair', 'counter', 'curtain', 'desk', 'door','otherfurniture']
            # new_data = {}
            # for key in keys_to_process:
            #     original_list = class2scans[key]
            #     sample_size = int(0.05 * len(original_list))
            #     random_sample = random.sample(original_list, sample_size)
            #     new_data[key] = random_sample
            # for key in new_data:
            #     class2scans[key] = new_data[key]
            # for class_name in __C.TYPE_WHITELIST:
            #     print('\t class_name: {0} | num of scans: {1}'.format(class_name, len(class2scans[class_name])))
            # csv_filename = '0.05scannet2.csv'
            # with open(csv_filename, mode='w', newline='') as csv_file:
            #     writer = csv.writer(csv_file)
    
            #     header = class2scans.keys()
            #     writer.writerow(header)
    
            #     max_data_length = max(len(class2scans[key]) for key in class2scans.keys())
            #     for i in range(max_data_length):
            #         row = [class2scans[key][i] if i < len(class2scans[key]) else '' for key in header]
            #         writer.writerow(row)
            # print(class2scans)
            selected_data = {}
            while True:
                # 步骤1：将数据转换成新字典的格式
                new_data = transform_data(class2scans)
                
                # 清空选择的数据
                selected_data = {}
                
                # 步骤2：随机选择5%的数据
                selected_data = select_random_samples(new_data, percent_to_select)
                
                # 步骤3：计算每个键对应的值的总和
                totals = calculate_totals(selected_data)
                
                # 步骤4：检查是否满足条件（每个键对应的值都大于0）
                if all(value > 0 for value in totals.values()):
                    break

            # 打印每个键对应的值的总和
            print(selected_data)
            # 初始化一个空的新字典，用于存储转换后的数据
            converted_data = {key: [] for key in class2scans.keys()}

            # 遍历最终生成的字典
            for sample, values in selected_data.items():
                # 遍历每个键
                for key, value in values.items():
                    # 如果值为1，将该样本添加到相应的键中
                    if value == 1:
                        converted_data[key].append(sample)
            converted_data['sofa']=class2scans['sofa']
            converted_data['table']=class2scans['table']
            converted_data['toilet']=class2scans['toilet']
            converted_data['window']=class2scans['window']

            # 打印转换后的数据
            print(converted_data)        
            with open("0.05scannet_scene", 'wb') as f:
                pickle.dump(converted_data, f, pickle.HIGHEST_PROTOCOL)
            for class_name in __C.TYPE_WHITELIST:
                print('\t class_name: {0} | num of scans: {1}'.format(class_name, len(converted_data[class_name])))


    else:
        class2scans = {c: [] for c in __C.TYPE_WHITELIST}
        all_scan_names = list(set([os.path.basename(x)[0:12] \
                                   for x in os.listdir(os.path.join(data_path, 'scannet_%s_detection_data' %split)) \
                                   if x.startswith('scene')]))
        for scan_name in all_scan_names:
            bboxes = np.load(os.path.join(data_path, 'scannet_%s_detection_data' %split, scan_name)+'_bbox.npy')
            label_ids = bboxes[:,-1]
            unique_label_ids = np.unique(label_ids)
            for nyuid in unique_label_ids:
                nyuid = int(nyuid)
                if nyuid in __C.NYU40IDS:
                    class_name = __C.TYPE_WHITELIST[np.where(__C.NYU40IDS==nyuid)[0][0]]
                    class2scans[class_name].append(scan_name)

        print('==== Split: %s | class to scans mapping is done ====' %split)
        for class_name in __C.TYPE_WHITELIST:
            print('\t class_name: {0} | num of scans: {1}'.format(class_name, len(class2scans[class_name])))

        #save class2scans to file...
        with open(class2scans_file, 'wb') as f:
            pickle.dump(class2scans, f, pickle.HIGHEST_PROTOCOL)
    return class2scans

if __name__ == '__main__':
    get_class2scans('../scannet/', split='val')
