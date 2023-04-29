import shutil
import datetime
import os

def move_logs_and_model(test_accuracy):

    # 获取当前日期时间
    now = datetime.datetime.now()

    date_time = now.strftime("%Y-%m-%d_%H-%M-%S")

    # 作为文件目录名
    directory_name = os.path.join("./model_logs", date_time)

    logs_and_model_path = directory_name + '_' + str(test_accuracy)
    
    # 创建存放模型和日志的文件夹
    os.makedirs(logs_and_model_path)
    
    # 移动模型文件
    shutil.move('./best_model.h5', logs_and_model_path + '/best_model.h5')
    
    # 移动CSV日志
    shutil.move('./training_logs.csv', logs_and_model_path + '/training_logs.csv')
    
    # 移动TensorBoard日志
    shutil.move('./logs', logs_and_model_path + '/logs')