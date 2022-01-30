import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader 
from torchvision import transforms
import os
import cv2
import matplotlib.pyplot as plt
import tqdm.notebook as tq
import torchvision.models as models
from tensorflow.keras.models import Sequential

epochs       = 5
batch_size   = 32
width        = 224
height       = 224

class CustomDataset(DataLoader):
    def __init__(self, DATA_PATH):
        # super(CustomDataset, self).__init__()
        labels_dict = ['0','1','2','3','4']
        self.images = []
        self.labels = []
        for i in labels_dict : # data chia anh theo tung label
          for image_file in os.listdir(DATA_PATH+'/'+i):
            image_path = os.path.join(DATA_PATH+'/'+i, image_file)
            self.images.append(image_path)
            self.labels.append(int(image_file.split(".")[1]))
    # (N, W, H, C)
    # (N, C, W, H)
    # RGB 190 85 10 => 190/255
    def __getitem__(self, idx):
        image = cv2.imread((self.images[idx]))
        image = cv2.resize(image, (width, height))
        label = self.labels[idx]
        # augmentation:
        aug = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]) # W H C -> C W H 
        image = aug(image)
        return image, label
    
    def __len__(self) -> int:
        return len(self.images)

if __name__ == "__main__":
    dataset = CustomDataset("/content/drive/MyDrive/Colab Notebooks/Cu_Chi_Tay/data/train")
    print(dataset.__getitem__(354)[1])
    print(dataset.__getitem__(354)[0].shape)
    print(dataset.__len__())
path_train = ''
path_valid = ''
trainDataset = CustomDataset(path_train)
trainLoader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True, num_workers=2)
 
validDataset = CustomDataset(path_valid)
validLoader = DataLoader(validDataset, batch_size=batch_size, shuffle=True, num_workers=2)

class MobileNetv2(nn.Module):
    def __init__(self) -> None:
        super(MobileNetv2, self).__init__()
        self.mobilenetv2 = models.mobilenet_v2(pretrained=True)
        self.mobilenetv2.classifier[1] = nn.Linear(1280, 5)
        self.softmax = nn.Sequential(nn.Softmax(dim=1))

    def forward(self, x):
      x = self.mobilenetv2(x)
      return x
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = MobileNetv2().to(device)
    dummy = torch.ones((1, 3, 224, 224)).to(device)
    model.eval() #non-gradient
    print(model(dummy))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print( device )
model = MobileNetv2()

optimizer = optim.SGD(model.parameters(), lr=3e-2)
loss_fn = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

#train
losses_train = []
losses_valid = []
accu_valid = []
for epoch in range(epochs):
  running_train_loss = 0.0
  running_valid_loss = 0.0
  running_valid_accu = 0.0

  
  model.train() #gradient optimize
  for img, label in tq.tqdm(trainLoader):
    
    optimizer.zero_grad()
    #img = img.to(device)
    lable = label.to(device)
    #print(label)
    output = model(img)
    #print(output)
    #output = output.to(device)
    loss = loss_fn(output, label)
    #optimizer.zero_grad()     
    loss.backward()
    optimizer.step()
    running_train_loss += loss.item() * img.size(0) 
  epoch_train_loss = running_train_loss / len(trainLoader)
  losses_train.append(epoch_train_loss)
  print("")
  print('Training, Epoch {} - Loss {}'.format(epoch+1, epoch_train_loss))

  model.eval()

  for (input, label) in tq.tqdm(validLoader): #for colab
    
    label2 = label.clone()
    label2 = label2.cpu().numpy()
    with torch.no_grad():
      output = model(input)

    pred_label = torch.argmax(output, dim=1)
    pred_label = pred_label.cpu().numpy()

    accuracy = np.count_nonzero(pred_label == label2) / batch_size
    running_valid_accu += accuracy

    loss = loss_fn(output, label)
    running_valid_loss += loss.item() * input.size(0)

    epoch_valid_loss = running_valid_loss / len(validLoader)
    epoch_valid_accu = running_valid_accu / len(validLoader)
    accu_valid.append(epoch_valid_accu)
    scheduler.step()
    losses_valid.append(epoch_valid_loss)

  print("")
  print('Validated, Epoch {} - Loss {} - Acc {}'.format( epoch+1, epoch_valid_loss, epoch_valid_accu))

def image_transform(img):
  transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  ])
  img = transform(img)
  return img
def predict(path):
    model.eval()
    img=cv2.imread(path)
    img = cv2.resize(img, (width, height))
    input=image_transform(img)
    #with torch.no_grad():
    output = model(input.unsqueeze(0))
    prob, pred_label = torch.max(output, dim=1)
    pred_label = pred_label.cpu().numpy()
    pred = pred_label[0]
    return pred

path_test = ''
lst=os.listdir(path_test)
lst.sort()
res=[]
for path_img in lst :
  res.append(predict(path_test + path_img))
res