import sys
sys.path.append('..')
import numpy as np
import pickle
import time
import scipy.misc
import platform

def make_generator(classname, image_size, batch_size, mode):
    if 'linux' in platform.platform().lower():
        filepath = '/home/yumin/dataset/web1000/%dimages_%s.pickle'%(image_size, classname)
    else:
        filepath = 'D:/v-yusuh/dataset/web1000/%dimages_%s.pickle'%(image_size, classname)
    with open(filepath, 'rb') as f:
        tmp = pickle.load(f, encoding='latin1') # list. images[i]: (h, w, 3)
    images, sel = tmp['data'], np.array([int(v) for v in tmp[mode]])
    images = np.array(images).transpose(0,3,1,2)   # (nbatch, 3*h*w)
    images = images[sel,:,:,:]
    H = images.shape[2]
    W = images.shape[3]
    print(images.shape)

    def get_epoch():
        np.random.shuffle(images)
        images_aug = np.copy(images).transpose(0,2,3,1)
        #print("images_aug shape")
        #print(images_aug.shape)
        for i in range(len(images_aug)): 
            # flip
            if np.random.uniform() > 0.5:
                images_aug[i,:,:,:] = np.fliplr(images_aug[i,:,:,:])
            # crop
            top = int(H*np.random.uniform(0,0.05))
            bottom = H - int(H*np.random.uniform(0,0.05)) 
            left = int(W*np.random.uniform(0,0.05))
            right = W - int(W*np.random.uniform(0,0.05))
            #print("top, bottom, left, right")
            #print(top, bottom, left, right)
            crop = np.copy(images_aug[i,top:bottom,left:right,:])
            #print("crop shape")
            #print(crop.shape)
            images_aug[i,:,:,:] = scipy.misc.imresize(crop, (H,W))

        images_aug = images_aug.transpose(0,3,1,2)
        for i in range(int(np.floor(len(images) / batch_size))):
            imgs_aug = np.copy(images_aug[i*batch_size:(i+1)*batch_size])

            yield (imgs_aug, )
    return get_epoch

def load(classname, batch_size=64, image_size=64):
    return (
        make_generator(classname, image_size, batch_size, 'train'),
        make_generator(classname, image_size, batch_size, 'val')
    )

if __name__ == '__main__':
    train_gen, valid_gen = load(64)
    t0 = time.time()
    for i, batch in enumerate(train_gen(), start=1):
        print("{}\t{}".format(str(time.time() - t0), batch[0][0,0,0,0]))
        if i == 1000:
            break
        t0 = time.time()
