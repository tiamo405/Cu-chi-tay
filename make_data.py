'''
cách lấy data :
b1 : cài thư viện opencv  bằng cách bật cmd rồi gõ pip install opencv-python
b2 : thay đổi label nếu bạn muốn lấy những số nào, sẽ hơi mỏi tay nếu lấy 1 lúc 10 số
b3 : name thay bằng tên của bạn
b4 : dùng tay phải để sát camera, giơ theo 1 bức ảnh nhỏ trên camera 

sẽ có 1 cửa sổ bật lên bằng camera của bạn nên hãy setup nếu không muốn gương mặt vào ảnh
chương trình sẽ tự chụp ảnh nên bạn chỉ cần giơ tay theo đúng là được
Rất cảm ơn vì đã giúp mình
Choco Tiamo <3
'''
import cv2
import os
if not os.path.exists('data/') : 
    os.mkdir('data/')
labels=["0","1","2","3","4"]
name="Nam"
for label in labels :
    print("make_data so ", label)
    i=0
    cap=cv2.VideoCapture(0)
    while(True):
        i+=1
        ret,frame =cap.read()
        if not ret :
            continue
        frame =cv2.resize(frame,dsize=None,fx=0.5,fy=0.5)
        cv2.imshow('camera',frame)

        if i >=100 :
            
            j=int(i-100)
            print("So anh capture" , j )
            if not os.path.exists('data/train') : 
                os.mkdir('data/train')
            if not os.path.exists('data/train'+'/'+label) : 
                os.mkdir('data/train'+'/'+label)
            cv2.imwrite('data/train/'+label + '/' + name + '.' + label + '.' + str(j) + '.png', frame)
        if i>=299 :
            break
        if cv2.waitKey(1) & 0xFF == 27: #ấn ESC để out
            break

labels=["0","1","2","3","4"]
name="Nam"
for label in labels :
    print("make_data so ", label)
    i=0
    cap=cv2.VideoCapture(0)
    while(True):
        i+=1
        ret,frame =cap.read()
        if not ret :
            continue
        frame =cv2.resize(frame,dsize=None,fx=0.5,fy=0.5)
        cv2.imshow('camera',frame)

        if i >=100 :
            
            j=int(i-100)
            print("So anh capture" , j )

            if not os.path.exists('data/valid') : 
                os.mkdir('data/valid')
            if not os.path.exists('data/valid'+'/'+label) : 
                os.mkdir('data/valid'+'/'+label)
            cv2.imwrite('data/valid/'+label + '/' + name + '.' + label + '.' + str(j) + '.png', frame)
        if i>=145 :
            break
        if cv2.waitKey(1) & 0xFF == 27: #ấn ESC để out
            break
cap.release()
cv2.destroyAllWindows()
