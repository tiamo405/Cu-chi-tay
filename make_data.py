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
'''
nó bị overfit, nên tôi đã bỏ mấy dòng để hiện ảnh nhỏ nhỏ đi, k biết có phải do nó không nữa.
'''
import cv2
import os
labels=["0","1","2","3","4","5"]
#labels=["9"]
name="Nam1"
for label in labels :
    print("make_data so ", label)
    i=0
    cap=cv2.VideoCapture(0)
    #image_hand=cv2.imread("image_hand\\"+label+'.png')
    while(True):
        i+=1
        ret,frame =cap.read()
        if not ret :
            continue
        frame =cv2.resize(frame,dsize=None,fx=0.5,fy=0.5)

        # cho 1 ảnh lên camera, ảnh có 3 chiều nha
        #h, w, c=image_hand.shape
        #frame[0: h, 0: w]=image_hand
        cv2.imshow('camera',frame)

        if i >=100 :
            
            j=int(i-100)
            print("So anh capture" , j )
            # tạo folder train nếu chưa có
            if not os.path.exists('train/') : 
                os.mkdir('train/')
            # lưu ảnh trên camera
            cv2.imwrite('train/' + name + '.' + label + '.' + str(j) + '.png', frame)
        if cv2.waitKey(1) & 0xFF == ord ('q'):
            break
        if i>=299 :
            break
cap.release()
cv2.destroyAllWindows()

