import sys
sys.path.append('..')
import numpy as np
import pickle
import time

def make_generator(classname, image_size, batch_size, mode):
    # filepath = '/home/yumin/codes/dcgan_code/Data/web1000/%dimages_%s.pickle'%(image_size, classname)
    filepath = 'D:/v-yusuh/dataset/web1000/%dimages_%s.pickle'%(image_size, classname)
    with open(filepath, 'rb') as f:
        tmp = pickle.load(f, encoding='latin1') # list. images[i]: (h, w, 3)
    images, sel = tmp['data'], np.array([int(v) for v in tmp[mode]])
    images = np.array(images).transpose(0,3,1,2)   # (nbatch, 3*h*w)
    images = images[sel,:,:,:]
    print(images.shape)

    def get_epoch():
        np.random.shuffle(images)
        for i in range(int(np.floor(len(images) / batch_size))):
            yield (np.copy(images[i*batch_size:(i+1)*batch_size]), )
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
