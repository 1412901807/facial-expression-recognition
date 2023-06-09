#该代码功能为对数据集进行分类
#数据路径 database存放csv文件 dataset存放按照训练集、验证集、测试集分成的不同的csv文件

import os 
import csv

database_path = 'data/'
datasets_path = 'data/'
csv_file = database_path+'fer2013.csv'
train_csv = datasets_path+'train.csv'
val_csv = datasets_path+'val.csv'

with open(csv_file) as f:
    csvr = csv.reader(f)
    #提取表头并存储
    header = next(csvr)
    print(header)
    #提取所有行
    rows = [row for row in csvr]
    #取出所有被标记为训练集的行并和表头一起存到对应csv文件中
    trn = [row[:-1] for row in rows if row[-1] == 'Training']
    csv.writer(open(train_csv, 'w+'), lineterminator='\n').writerows([header[:-1]] + trn)
    print(len(trn))
    #取出所有被标记为验证集的行并和表头一起存到对应csv文件中
    val = [row[:-1] for row in rows if row[-1] != 'Training']
    csv.writer(open(val_csv, 'w+'), lineterminator='\n').writerows([header[:-1]] + val)
    print(len(val))