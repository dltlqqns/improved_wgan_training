import sys
sys.path.append('..')
import os
import glob
import scipy.misc
import pickle
from utils import mkdir_p, get_image
import numpy as np
import platform
import pandas as pd

DATASET = 'CUB_200_2011' #'web1000'
CLASSNAME = '' #'truck'
IS_CROP = True #True
IMG_SIZE = 64 #128 
CIFAR_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
if platform.system()=='Linux':
    ROOT_DIR = '/home/yumin/dataset/%s/'%DATASET
else:
    ROOT_DIR = 'D:/v-yusuh/dataset/%s/'%DATASET
OUT_DIR = 'data/%s'%DATASET

# copied from StackGAN/misc/preprocess_bird.py
def load_bbox(data_dir):
    bbox_path = os.path.join(data_dir, 'bounding_boxes.txt')
    df_bounding_boxes = pd.read_csv(bbox_path,
                                    delim_whitespace=True,
                                    header=None).astype(int)
    #
    filepath = os.path.join(data_dir, 'images.txt')
    df_filenames = pd.read_csv(filepath, delim_whitespace=True, header=None)
    filenames = df_filenames[1].tolist()
    print('Total filenames: ', len(filenames), filenames[0])
    #
    filename_bbox = {img_file[:-4]: [] for img_file in filenames}
    numImgs = len(filenames)
    for i in range(0, numImgs):
        # bbox = [x-left, y-top, width, height]
        bbox = df_bounding_boxes.iloc[i][1:].tolist()

        key = filenames[i][:-4]
        filename_bbox[key] = bbox
    #
    return filename_bbox

def convert_dataset_pickle(root_dir, dataset, classname, img_size):
    #out_dir = os.path.join(ROOT_DIR)
    out_dir = OUT_DIR
    mkdir_p(out_dir)
    print("save dataset to %s"%out_dir)

    if 'cifar' in dataset.lower():
        print("CIFAR-10!!")
        print(classname)
        imgs = np.empty(shape=[0,3072])
        labels = []
        for i in range(1,6):
            filename = 'data_batch_%d'%i
            print(filename)
            with open(os.path.join(root_dir, filename),'rb') as f:
                tmp = pickle.load(f)
                imgs = np.concatenate((imgs, tmp['data']), axis=0)
                labels = labels + tmp['labels']
        class_id = CIFAR_CLASSES.index(CLASSNAME)
        sel = np.array([i for i,x in enumerate(labels) if x==class_id], dtype=np.int32)
        print(sel.shape)
        imgs = imgs[sel]
        imgs = imgs.reshape(len(sel),3,32,32).transpose((0,2,3,1))
        lr_images = np.array([scipy.misc.imresize(img, [img_size, img_size], 'bicubic') for img in imgs])

    elif 'web' in dataset.lower():
        print("Web!!")
        filenames = glob.glob(os.path.join(root_dir, classname, '*.jpg'))
        print("#img: " + str(len(filenames)))

        imgs = []
        for filename in filenames:
            img = scipy.misc.imread(filename)
            img = img.astype('uint8')
            img = scipy.misc.imresize(img, [img_size, img_size], 'bicubic')
            imgs.append(img)
    elif 'cub' in dataset.lower():
        print("CUB!!")
        filepath = os.path.join(root_dir, 'images.txt')
        df_filenames = pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        print("#img: " + str(len(filenames)))
        filename_bbox = load_bbox(root_dir)

        imgs = []
        for filename in filenames:
            key = filename[:-4]
            #print(key)
            bbox = filename_bbox[key]
            img = get_image(os.path.join(root_dir, 'images', filename), img_size, is_crop=IS_CROP, bbox=bbox)
            img = img.astype('uint8')
            imgs.append(img)
    else:
        error("unknown dataset!!")
        

    # train/val division
    randperm = np.random.permutation(len(imgs))
    num_use = 3000 #len(imgs)
    num_train = int(np.floor(num_use*0.9))
    train = randperm[:num_train]
    val = randperm[num_train:num_use]
    # Save images to .pickle
    if IS_CROP:
        outfile = os.path.join(out_dir, 'medium_%dimages_%s_crop.pickle'%(img_size, classname))
    else:
        outfile = os.path.join(out_dir, 'medium_%dimages_%s.pickle'%(img_size, classname))
    with open(outfile, 'wb') as f_out:
        pickle.dump({'data': imgs, 'train': train, 'val': val}, f_out)
    

if __name__ == '__main__':
    convert_dataset_pickle(ROOT_DIR, DATASET, CLASSNAME, IMG_SIZE)
