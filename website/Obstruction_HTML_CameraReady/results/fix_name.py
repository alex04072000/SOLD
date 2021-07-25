import numpy as np
import cv2
import os
import shutil

"""Fence"""
# base_folder = r'C:\Users\user\Documents\Alex\Research\VideoReflectionRemoval\SupplementaryMaterial_CameraReady\results\fence'
# base_folder_new = base_folder+'_new'
# method_old = ['0', '1', '2']
# method_new = ['Xue2015', 'Du2018', 'Ours']
# number = 11
# dataset_new = ['Xue et al.']*6
# dataset_new.extend(['Du et al.']*5)
# idx_new = list(range(6))
# idx_new.extend(range(5))
# assert len(dataset_new) == number

"""Reflection"""
base_folder = r'C:\Users\user\Documents\Alex\Research\VideoReflectionRemoval\SupplementaryMaterial_CameraReady\results\reflection'
base_folder_new = base_folder+'_new'
method_old = ['single0', 'single1', 'single2', 'single3', 'single4', 'single5', 'multi0', 'multi1', 'multi2', 'multi3', 'multi4', 'multi5']
method_new = ['Fan2017', 'Zhang2018', 'Yang2018', 'Wei2019', 'Jin2018', 'Gandelsman2019', 'Sinha2012', 'Li2013', 'Guo2014', 'Xue2015', 'Alayrac2019', 'Ours']
number = 26
dataset_new = ['Xue et al.']*10
dataset_new.extend(['Li and Brown']*8)
dataset_new.extend(['Guo et al.']*4)
dataset_new.extend(['Kopf et al.']*4)
idx_new = list(range(10))
idx_new.extend(range(8))
idx_new.extend(range(4))
idx_new.extend(range(4))
assert len(dataset_new) == number

for method_idx, method in enumerate(method_old):
    for idx in range(number):
        if idx == 21:
            continue
        if not os.path.exists(os.path.join(base_folder_new, dataset_new[idx])):
            os.makedirs(os.path.join(base_folder_new, dataset_new[idx]))

        B_path = os.path.join(os.path.join(base_folder, method), str(idx).zfill(5)+'_B2.jpg')
        B = cv2.imread(B_path)
        cv2.imwrite(os.path.join(os.path.join(base_folder_new, dataset_new[idx]), str(idx_new[idx]).zfill(5)+'_Background_'+method_new[method_idx]+'.png'), B)

        F_path = os.path.join(os.path.join(base_folder, method), str(idx).zfill(5)+'_F2.jpg')
        F = cv2.imread(F_path)
        cv2.imwrite(os.path.join(os.path.join(base_folder_new, dataset_new[idx]), str(idx_new[idx]).zfill(5)+'_Reflection_'+method_new[method_idx]+'.png'), F)

"""input"""
for idx in range(number):
    if idx == 21:
        continue
    shutil.copyfile(os.path.join(os.path.join(base_folder, 'Input'), str(idx).zfill(5)+'_I.mp4'), os.path.join(os.path.join(base_folder_new, dataset_new[idx]), str(idx_new[idx]).zfill(5)+'_Input.mp4'))

    B_path = os.path.join(os.path.join(base_folder, 'Input'), str(idx).zfill(5)+'_I2.jpg')
    B = cv2.imread(B_path)
    cv2.imwrite(os.path.join(os.path.join(base_folder_new, dataset_new[idx]), str(idx_new[idx]).zfill(5)+'_Input.png'), B)