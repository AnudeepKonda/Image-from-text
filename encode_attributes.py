import re
import os
import numpy as np

img_names = []
encoding = []
file_path = os.path.dirname(os.path.abspath(__file__)) + "\\pubfig_imgs"
with open("pubfig_attributes.txt") as img_file:
    count = 0
    print("Starting to fetch attributes")
    count = 0
    for line in img_file:
        arr = re.split(' |\t', line)
        if arr[0] == '#':
            continue
        img_name = arr[0]
        num = 0
        for i in range(1,20):
            try:
                test = int(arr[i])
                img_name += '-'+arr[i]
                num = i
                break
            except:
                img_name += '-'+arr[i]
        img_attr = arr[num:]
        for item in img_attr:
            item = float(item)
            if item <= 0:
                item = 0
            else:
                item = 1
        img_names.append(img_name)
        encoding.append(img_attr)
        count += 1
        if count % 10000 == 0:
            print("10000 more done")

encoding = np.asarray(encoding)
img_names = np.asarray(img_names)