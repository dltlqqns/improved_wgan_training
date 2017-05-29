import sys
sys.path.append('..')
import os
import glob
import scipy.misc
import pickle
from utils import mkdir_p
import numpy as np
import platform

DATASET = 'web1000'
CLASSNAME = 'truck'
IMG_SIZE = 128
CIFAR_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
if platform.system()=='Linux':
    ROOT_DIR = '/home/yumin/dataset/%s/'%DATASET
else:
    ROOT_DIR = 'D:/v-yusuh/dataset/%s/'%DATASET

def convert_dataset_pickle(root_dir, dataset, classname, img_size):
    out_dir = os.path.join(ROOT_DIR)
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

    else:
        print("Web!!")
        filenames = glob.glob(os.path.join(root_dir, classname, '*.jpg'))
        print("#img: " + str(len(filenames)))

        imgs = []
        for filename in filenames:
            img = scipy.misc.imread(filename)
            img = img.astype('uint8')
            img = scipy.misc.imresize(img, [img_size, img_size], 'bicubic')
            imgs.append(img)

    # train/val division
    outfile = os.path.join(out_dir, '%dimages_%s_trainval.pickle'%(img_size, classname))
    randperm = np.random.permutation(len(imgs))
    num_train = int(np.floor(len(imgs)*0.9))
    train = randperm[:num_train]
    val = randperm[num_train:]
    # Save images to .pickle
    outfile = os.path.join(out_dir, '%dimages_%s.pickle'%(img_size, classname))
    with open(outfile, 'wb') as f_out:
        pickle.dump({'data': imgs, 'train': train, 'val': val}, f_out)
    

if __name__ == '__main__':
    convert_dataset_pickle(ROOT_DIR, DATASET, CLASSNAME, IMG_SIZE)
