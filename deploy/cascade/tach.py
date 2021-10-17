import os
import shutil
# đường dẫn lấy ảnh
imgs_dir_mask = r'D:\TruongCongHau\KhoaLuanTotNghiep_2021\khoaluantotnghiep\deploy\images\du_lieu_sinh_vien\374_09\test'
imgs_out = r'D:\TruongCongHau\KhoaLuanTotNghiep_2021\khoaluantotnghiep\deploy\images\du_lieu_sinh_vien\374_09\test_mask'
# chuoi - > tách mask 
folders = os.listdir(imgs_dir_mask)  # data
for folder in folders:
    os.mkdir(imgs_out + '\\' + folder)
    imgs = os.listdir(os.path.join(imgs_dir_mask, folder))
    # print(len(imgs))
    for i in range(len(imgs)):
        str = imgs[i]
        arr = str.split('_')
        print(len(arr))
        flag = False
        leng = len(arr) - 1
        for ix in range(leng):
            kq = arr[ix].isnumeric()
            if kq == False:
                flag = True
                break
        if flag is False:
            out = imgs_out + '\\' + folder + '\\' + imgs[i]
            img = imgs_dir_mask + '\\' + folder + '\\' + imgs[i]
            shutil.copy(img, out)

#  tách no mask
# folders = os.listdir(imgs_dir_mask)  # data
# for folder in folders:
#     os.mkdir(imgs_out + '\\' + folder)
#     imgs = os.listdir(os.path.join(imgs_dir_mask, folder))
#     # print(len(imgs))
#     for i in range(len(imgs)):
#         str = imgs[i]
#         arr = str.split('_')
#         print(len(arr))
#         flag = False
#         leng = len(arr) - 1
#         for ix in range(leng):
#             kq = arr[ix].isnumeric()
#             if kq == False:
#                 flag = True
#                 break
#         if flag is True:
#             out = imgs_out + '\\' + folder + '\\' + imgs[i]
#             img = imgs_dir_mask + '\\' + folder + '\\' + imgs[i]
#             shutil.copy(img, out)
