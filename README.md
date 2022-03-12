# cu-chi-tay

* Lời nói đầu : Đây là prj làm trong thời gian nghỉ tết, nội dung là dự đoán kí hiệu tay chuyển sang số ( từ 0 đến 4 )
* Dự định tương lai : cho 1 video vào là có thể xuất ra 1 câu của video đó
* Có thể bỏ qua bước 2,3 để đến bước 4

# 1.Cách cài đặt :
* Di chuyển terminal đến folder
* pip install -r requirements.txt ( cài đặt các thư viện cần dùng, khuyến khích sử dụng anaconda )

# 2.Make data 
* python make_data.py
* Link ảnh cách để tay đúng với số : https://drive.google.com/drive/folders/1zm47smDT6d6CmT7co6IRNHf15PntASyS?usp=sharing ( ở bìa này mình chỉ dùng các số từ 0 đến 4 )

# 3.Train model :
* python train.py

# 4.Test code :
* python test_torch.py
* Có thể bỏ qua bước 2 và 3 để đến bước 4 vì mình đã train trước và lưu model rồi.

# 5.Mentors :
* QuangTran
# 6.Ngoài lề :
* https://colab.research.google.com/drive/102dEpDeqyVCj5Mmj-1LJQ7vapDae08wv?usp=sharing
* link colab bài model, có 1 chút khác với các file trên nhưng model em lưu từ đây về rồi mới cho chạy trên laptop cá nhân
