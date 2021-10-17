# mask-----------------------------
import os
import shutil
# đường dẫn lấy ảnh
imgs_dir_mask = r'C:\Users\CongHau\Desktop\Data_GuiThayHa\DATA_374\AFDB_masked_face_dataset'
imgs_out = r'C:\Users\CongHau\Desktop\Data_GuiThayHa\DATA_374\mask'
# mask
folders = os.listdir(imgs_dir_mask)  # data
for folder in folders:
    imgs = os.listdir(os.path.join(imgs_dir_mask, folder))
    os.mkdir(imgs_out + '\\' + folder)
    if len(imgs) > 5:
        for i in range(5):
            img_out = imgs_out + '\\' + folder
            img = imgs_dir_mask + '\\' + folder + '\\' + imgs[i]
            shutil.copy(img, img_out)

    else:
        for i in range(len(imgs)):
            img_out = imgs_out + '\\' + folder
            img = imgs_dir_mask + '\\' + folder + '\\' + imgs[i]
            shutil.copy(img, img_out)
# nomask-----------------------------
