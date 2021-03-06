# coding:utf-8
import glob
import os.path
import numpy as np
import os
import re

import random

INPUT_DATA = './4sv_mtcnn/' 
pairs_file_path = './val/pairs.txt'

folders = os.listdir(INPUT_DATA)
id_nums = len(folders)
def produce_same_pairs():
    matched_result = []
    for j in range(3000):
        id_int= random.randint(0,id_nums-1)

        id_dir = os.path.join(INPUT_DATA, folders[id_int])

        id_imgs_list = os.listdir(id_dir)

        id_list_len = len(id_imgs_list)

        id1_img_file = id_imgs_list[random.randint(0,id_list_len-1)]
        id2_img_file = id_imgs_list[random.randint(0,id_list_len-1)]

        id1_path = os.path.join(id_dir, id1_img_file)
        id2_path = os.path.join(id_dir, id2_img_file)

        same = 1
        #print([id1_path + '\t' + id2_path + '\t',same])
        matched_result.append((id1_path + ',' + id2_path + ',',same))
    return matched_result


def produce_unsame_pairs():
    unmatched_result = []  
    for j in range(3000):
        id1_int = random.randint(0,id_nums-1)
        id2_int = random.randint(0,id_nums-1)
        while id1_int == id2_int:
            id1_int = random.randint(0,id_nums-1)
            id2_int = random.randint(0,id_nums-1)

        id1_dir = os.path.join(INPUT_DATA, folders[id1_int])
        id2_dir = os.path.join(INPUT_DATA, folders[id2_int])

        id1_imgs_list = os.listdir(id1_dir)
        id2_imgs_list = os.listdir(id2_dir)
        id1_list_len = len(id1_imgs_list)
        id2_list_len = len(id2_imgs_list)

        id1_img_file = id1_imgs_list[random.randint(0, id1_list_len-1)]
        id2_img_file = id2_imgs_list[random.randint(0, id2_list_len-1)]

        id1_path = os.path.join(id1_dir, id1_img_file)
        id2_path = os.path.join(id2_dir, id2_img_file)

        unsame = 0
        unmatched_result.append((id1_path + ',' + id2_path + ',',unsame))
    return unmatched_result


same_result = produce_same_pairs()
print('same record: %d'% len(same_result))
unsame_result = produce_unsame_pairs()
print('unsame record: %d'% len(unsame_result))

all_result = same_result + unsame_result

#random.shuffle(all_result)
#print(all_result)

file = open(pairs_file_path, 'w')
for line in all_result:
    file.write(line[0] + str(line[1]) + '\n')

file.close()
print('Successfully!')