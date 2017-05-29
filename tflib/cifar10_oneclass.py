import numpy as np

import os
import urllib
import gzip
import cPickle as pickle

CLASSNAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo)
    fo.close()
    return dict['data'], dict['labels']

def cifar_generator(classname, filenames, batch_size, data_dir):
    class_id = CLASSNAMES.index(classname)
    all_data = []
    all_labels = []
    for filename in filenames:
        tmp_data, tmp_labels = unpickle(data_dir + '/' + filename)
        all_data.append(tmp_data)
        all_labels.append(tmp_labels)

    images = np.concatenate(all_data, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    sel = labels==class_id
    images = images[sel]

    print(images.shape)
    print(labels.shape)

    def get_epoch():
        np.random.shuffle(images)

        for i in xrange(len(images) / batch_size):
            yield np.copy(images[i*batch_size:(i+1)*batch_size])

    return get_epoch


def load(classname, batch_size, data_dir):
    return (
        cifar_generator(classname, ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5'], batch_size, data_dir), 
        cifar_generator(classname, ['test_batch'], batch_size, data_dir)
    )
