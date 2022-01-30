from re import I
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import os
import cv2
import time
import torchvision.models as models

batch_size = 32
LR = 1e-3 
epochs = 10
width=224
height=224

class MobileNetv2(nn.Module):
    def __init__(self) -> None:
        super(MobileNetv2, self).__init__()
        self.mobilenetv2 = models.mobilenet_v2(pretrained=True)
        self.mobilenetv2.classifier[1] = nn.Linear(1280, 5)
        self.softmax = nn.Sequential(nn.Softmax(dim=1))

    def forward(self, x):
      x = self.mobilenetv2(x)
      return x

def image_transform(img):
  transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  ])
  img = transform(img)
  return img

def predict(PATH_img):
    img=cv2.imread(PATH_img)
    img = cv2.resize(img, (width, height))
    image=image_transform(img)
    input = image.unsqueeze(0)
    model.eval()

    with torch.no_grad():
        output = model(input)
    prob, pred_label = torch.max(output, dim=1)
    pred_label = pred_label.cpu().numpy()
    pred = pred_label[0]
    return pred

model = MobileNetv2()
model.load_state_dict(torch.load('Mobilenet_cu_chi_tay.pth'))
'''
PATH_img='data\\test\\test_folder\\05.png'
res=predict(PATH_img)
print(res)
'''
if not os.path.exists('test_camera/') : 
    os.mkdir('test_camera/')
i = 0
cap=cv2.VideoCapture(0)
time_2= time.time()
ans=[]
while True :
    ret,frame =cap.read()
    if not ret :
        continue
    frame =cv2.resize(frame,dsize=None,fx=0.5,fy=0.5)
    cv2.imshow('camera',frame)
    
    print("Giữ nguyên tư thế tay để chụp")

    #time.sleep(10)
    time_1= time.time()
    if time_1 - time_2 >=5 :
        i+=1
        cv2.imwrite('test_camera/' + 'camera' + '.' + str(i) + '.png', frame)
        print("đã chụp được ảnh")
        path_img='test_camera\\' + 'camera' + '.' + str(i)+ '.png'
        print('path_img : ',path_img)
        res=predict(path_img)
        print('Số dự đoán là :',res)
        ans.append(res)

        cv2.putText(frame,f"res index {i} : {int(res)}",(10,20), 
        cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),3)
        cv2.imshow('res',frame)
        time_2 = time_1
    if cv2.waitKey(1) & 0xFF == 27: #ESC để out
        break
cap.release()
cv2.destroyAllWindows()
print(ans)
