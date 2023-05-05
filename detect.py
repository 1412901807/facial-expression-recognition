import cv2
import time
import numpy as np
import torch
from model import vggnet,resnet
# 表情类别
classes=['anger','disgust','fear','happy','sad','surprised','normal']
# 加载模型
model = vggnet()
model.load_state_dict(torch.load('model/vggnet/vggnet.h5'))
model.eval()

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    # 读取摄像头图片
    ret, frame = cap.read()
    
    # 等待5秒
    time.sleep(5)
    
    # 转换成灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 调整大小
    resized = cv2.resize(gray, (48, 48))
    
    # 归一化
    img = resized / 255.
    
    # 转换成PyTorch的Tensor类型
    img_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
    
    # 使用模型进行预测
    output = model(img_tensor)
    pred = torch.argmax(output).item()
    
    # 打印预测结果
    print("Predicted class:", classes[pred])
    
    # 显示图片
    cv2.imshow('frame', resized)
    
    # 按下q键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()
