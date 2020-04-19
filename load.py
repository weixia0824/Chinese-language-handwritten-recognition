import pickle
import os
import glob
import numpy as np
from tqdm import tqdm
from PIL import Image

config = {
    "save": False
}
root = './character'
label_path = os.path.join(root, 'labels.txt')
char_dict = {}

# 此处一定要加上encoding='GB2312-80',否则默认utf-8解析会出错
with open(label_path, 'r', encoding='GB2312-80') as f:
    label_str = f.read()
    for i in range(len(label_str)):
        if i < 100:
            print(label_str[i])
        char_dict[int(i)] = label_str[i]

# 可以看到一共有labels有6825个字符，与解析vectors文件得到的字符数是相同的
print("number of char: ", len(char_dict))
with open('char_dict_HIT-OR3C', 'wb') as f:
    pickle.dump(char_dict, f)

# Process images
dataset = glob.glob(os.path.join(root, '*_images'))
save_dir = "sample_dataset"
 
for file_id in tqdm(range(len(dataset))):
    path = dataset[file_id]
    with open(path, 'rb') as f:

        nChars = int(np.fromfile(f, dtype='int32', count=1)[0])
        nCharPixelHeight = int(np.fromfile(f, dtype='uint8', count=1)[0])
        nCharPixelWidth = int(np.fromfile(f, dtype='uint8', count=1)[0])

        for n in range(nChars):

            # get image
            img = np.fromfile(f, dtype='uint8', count=nCharPixelWidth * nCharPixelHeight)
            img = img.reshape(nCharPixelWidth, nCharPixelHeight)

            # save image
            if config["save"]:
                label = char_dict[n]
                save_name = label.strip() + '.jpg'  # chinese encoding used in filename
                image = Image.fromarray(img)
                image.save(os.path.join(save_dir, save_name))
