from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import os
import cv2
'''
chú ý chạy ở máy thì path có thể sẽ khác nha, k dùng \ mà dùng // , 
1 / nhiều khi lỗi do code tính /t /n là tab hay xuống dòng
'''
def aug_rotate(path,save): # xoay ảnh
  i=0
  for name_img in os.listdir(path) :
    
    img=load_img(path+'/'+name_img)
    img=img_to_array(img)

    data = expand_dims(img, 0)
    # xoay ảnh 45 độ
    myImageGen = ImageDataGenerator(rotation_range=45)
    # Batch_Size= 1 -> mỗi lần sinh ra 1 ảnh
    gen = myImageGen.flow(data, batch_size=1)
    for h in range(1): # lưu 1 ảnh
      myBatch = gen.next()
      image = myBatch[0].astype('uint8')

      # tạo tên mới cho ảnh gồm tên ng, lable, ảnh thứu bn, lần tạo mới thứ bn, tên def
      tmp=[]
      tmp=name_img.split('.') # tách name_img thành từng phần để tí tạo name mới
      new_name_img=""
      for j in range(0,len(tmp)-1) : # ví dụ tên ban đầu là nam.0.0.png thì new name là  nam.0.0 +ảnh thứ mấy + loại ảnh+png
        new_name_img=new_name_img+ tmp[j]+ '.'
      new_name_img=new_name_img+ str(i)+ '.'+ 'rotate.'+ tmp[len(tmp)-1]
      print(new_name_img)
      cv2.imwrite(save+'\\'+new_name_img, image)
    i+=1
  print("Đã lưu được : ",i,"ảnh")

path="train" 
save="test_agument" 
if not os.path.exists(save) : 
  os.mkdir(save)
aug_rotate(path,save)