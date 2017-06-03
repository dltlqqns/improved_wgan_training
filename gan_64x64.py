import os, sys
sys.path.append(os.getcwd())

import time
import functools

import numpy as np
import tensorflow as tf
import sklearn.datasets

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
#import tflib.small_imagenet
import tflib.oneclass
import tflib.ops.layernorm
import tflib.plot
import os

# Download 64x64 ImageNet at http://image-net.org/small/download.php and
# fill in the path to the extracted files here!
#DATA_DIR = ''
#if len(DATA_DIR) == 0:
#    raise Exception('Please specify path to data directory in gan_64x64.py!')

MODE = 'wgan-gp' # dcgan, wgan, wgan-gp, lsgan
DIM = 32 # Model dimensionality
CRITIC_ITERS = 5 # How many iterations to train the critic for
N_GPUS = 1 # Number of GPUs
BATCH_SIZE = 64 # Batch size. Must be a multiple of N_GPUS
ITERS = 200000 # How many iterations to train for
LAMBDA = 10 # Gradient penalty lambda hyperparameter
TARGET_SIZE = 64
OUTPUT_DIM = TARGET_SIZE*TARGET_SIZE*3 # Number of pixels in each iamge
CLASSNAME = '' #'truck'
ARCH = 'WGAN' #'ResNet' #
MSG = 'CUB_hidden32'
EXP_NAME = '{}{}_{}_{}_M{}'.format(ARCH, TARGET_SIZE, MODE, CLASSNAME, MSG)
if not os.path.exists('samples/{}'.format(EXP_NAME)):
	os.mkdir('samples/{}'.format(EXP_NAME))
if not os.path.exists('models/{}'.format(EXP_NAME)):
	os.mkdir('models/{}'.format(EXP_NAME))
LOAD_ITER = 0 #100000
LOAD_PATH = 'models/{}/model.ckpt-{}'.format(EXP_NAME, LOAD_ITER)
DATA_DIR = 'data/CUB_200_2011'
IS_CROP = True

lib.print_model_settings(locals().copy())

def GeneratorAndDiscriminator():
    """
    Choose which generator and discriminator architecture to use by
    uncommenting one of these lines.
    """

    # Baseline (G: DCGAN, D: DCGAN)
    #return DCGANGenerator, DCGANDiscriminator

    # Baseline 128 (G: DCGAN, D: DCGAN)
    #return DCGANGenerator_128, DCGANDiscriminator_128

    # Baseline (G: DCGAN, D: DCGAN) no critic BN
    return DCGANGenerator, functools.partial(DCGANDiscriminator, bn=False)
	
    # Baseline 128 (G: DCGAN, D: DCGAN) no critic BN
    #return DCGANGenerator_128, functools.partial(DCGANDiscriminator_128, bn=False)

    # BEGAN
    #return BEGANGenerator, functools.partial(DCGANDiscriminator_128, bn=False)

    # No BN and constant number of filts in G
    # return WGANPaper_CrippledDCGANGenerator, DCGANDiscriminator

    # 512-dim 4-layer ReLU MLP G
    # return FCGenerator, DCGANDiscriminator

    # No normalization anywhere
    # return functools.partial(DCGANGenerator, bn=False), functools.partial(DCGANDiscriminator, bn=False)

    # Gated multiplicative nonlinearities everywhere
    # return MultiplicativeDCGANGenerator, MultiplicativeDCGANDiscriminator

    # tanh nonlinearities everywhere
    # return functools.partial(DCGANGenerator, bn=True, nonlinearity=tf.tanh), \
    #        functools.partial(DCGANDiscriminator, bn=True, nonlinearity=tf.tanh)

    # 101-layer ResNet G and D
    #return ResnetGenerator, ResnetDiscriminator

    raise Exception('You must choose an architecture!')

DEVICES = ['/gpu:{}'.format(i) for i in range(N_GPUS)]

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs, initialization='he')
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs, initialization='he')
    return LeakyReLU(output)

def Batchnorm(name, axes, inputs):
    if ('Discriminator' in name) and (MODE == 'wgan-gp'):
        if axes != [0,2,3]:
            raise Exception('Layernorm over non-standard axes is unsupported')
        return lib.ops.layernorm.Layernorm(name,[1,2,3],inputs)
    else:
        return lib.ops.batchnorm.Batchnorm(name,axes,inputs,fused=True)

def pixcnn_gated_nonlinearity(a, b):
    return tf.sigmoid(a) * tf.tanh(b)

def SubpixelConv2D(*args, **kwargs):
    kwargs['output_dim'] = 4*kwargs['output_dim']
    output = lib.ops.conv2d.Conv2D(*args, **kwargs)
    output = tf.transpose(output, [0,2,3,1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0,3,1,2])
    return output

def ResidualBlock(name, input_dim, output_dim, filter_size, inputs, resample=None, he_init=True):
    """
    resample: None, 'down', or 'up'
    """
    if resample=='down':
        conv_shortcut = functools.partial(lib.ops.conv2d.Conv2D, stride=2)
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=int(input_dim/2))
        conv_1b       = functools.partial(lib.ops.conv2d.Conv2D, input_dim=int(input_dim/2), output_dim=int(output_dim/2), stride=2)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=int(output_dim/2), output_dim=output_dim)
    elif resample=='up':
        conv_shortcut = SubpixelConv2D
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=int(input_dim/2))
        conv_1b       = functools.partial(lib.ops.deconv2d.Deconv2D, input_dim=int(input_dim/2), output_dim=int(output_dim/2))
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=int(output_dim/2), output_dim=output_dim)
    elif resample==None:
        conv_shortcut = lib.ops.conv2d.Conv2D
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim,  output_dim=int(input_dim/2))
        conv_1b       = functools.partial(lib.ops.conv2d.Conv2D, input_dim=int(input_dim/2),  output_dim=int(output_dim/2))
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=int(input_dim/2), output_dim=output_dim)

    else:
        raise Exception('invalid resample value')

    if output_dim==input_dim and resample==None:
        shortcut = inputs # Identity skip-connection
    else:
        shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1,
                                 he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = tf.nn.relu(output)
    output = conv_1(name+'.Conv1', filter_size=1, inputs=output, he_init=he_init, weightnorm=False)
    output = tf.nn.relu(output)
    output = conv_1b(name+'.Conv1B', filter_size=filter_size, inputs=output, he_init=he_init, weightnorm=False)
    output = tf.nn.relu(output)
    output = conv_2(name+'.Conv2', filter_size=1, inputs=output, he_init=he_init, weightnorm=False, biases=False)
    output = Batchnorm(name+'.BN', [0,2,3], output)

    return shortcut + (0.3*output)

# ! Generators

def FCGenerator(n_samples, noise=None, FC_DIM=512):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = ReLULayer('Generator.1', 128, FC_DIM, noise)
    output = ReLULayer('Generator.2', FC_DIM, FC_DIM, output)
    output = ReLULayer('Generator.3', FC_DIM, FC_DIM, output)
    output = ReLULayer('Generator.4', FC_DIM, FC_DIM, output)
    output = lib.ops.linear.Linear('Generator.Out', FC_DIM, OUTPUT_DIM, output)

    output = tf.tanh(output)

    return output

def DCGANGenerator(n_samples, noise=None, dim=DIM, bn=True, nonlinearity=tf.nn.relu):
    lib.ops.conv2d.set_weights_stdev(0.02)
    lib.ops.deconv2d.set_weights_stdev(0.02)
    lib.ops.linear.set_weights_stdev(0.02)

    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*8*dim, noise)
    output = tf.reshape(output, [-1, 8*dim, 4, 4])
    if bn:
        output = Batchnorm('Generator.BN1', [0,2,3], output)
    output = nonlinearity(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.2', 8*dim, 4*dim, 5, output)
    if bn:
        output = Batchnorm('Generator.BN2', [0,2,3], output)
    output = nonlinearity(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.3', 4*dim, 2*dim, 5, output)
    if bn:
        output = Batchnorm('Generator.BN3', [0,2,3], output)
    output = nonlinearity(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.4', 2*dim, dim, 5, output)
    if bn:
        output = Batchnorm('Generator.BN4', [0,2,3], output)
    output = nonlinearity(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.5', dim, 3, 5, output)
    output = tf.tanh(output)

    lib.ops.conv2d.unset_weights_stdev()
    lib.ops.deconv2d.unset_weights_stdev()
    lib.ops.linear.unset_weights_stdev()

    return tf.reshape(output, [-1, OUTPUT_DIM])
	
def DCGANGenerator_128(n_samples, noise=None, dim=DIM, bn=True, nonlinearity=tf.nn.relu):
    lib.ops.conv2d.set_weights_stdev(0.02)
    lib.ops.deconv2d.set_weights_stdev(0.02)
    lib.ops.linear.set_weights_stdev(0.02)

    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*16*dim, noise)
    output = tf.reshape(output, [-1, 16*dim, 4, 4])
    if bn:
        output = Batchnorm('Generator.BN1', [0,2,3], output)
    output = nonlinearity(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.2', 16*dim, 8*dim, 5, output)
    if bn:
        output = Batchnorm('Generator.BN2', [0,2,3], output)
    output = nonlinearity(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.3', 8*dim, 4*dim, 5, output)
    if bn:
        output = Batchnorm('Generator.BN3', [0,2,3], output)
    output = nonlinearity(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.4', 4*dim, 2*dim, 5, output)
    if bn:
        output = Batchnorm('Generator.BN4', [0,2,3], output)
    output = nonlinearity(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.5', 2*dim, dim, 5, output)
    if bn:
        output = Batchnorm('Generator.BN5', [0,2,3], output)
    output = nonlinearity(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.6', dim, 3, 5, output)
    output = tf.tanh(output)

    lib.ops.conv2d.unset_weights_stdev()
    lib.ops.deconv2d.unset_weights_stdev()
    lib.ops.linear.unset_weights_stdev()

    return tf.reshape(output, [-1, OUTPUT_DIM])


def WGANPaper_CrippledDCGANGenerator(n_samples, noise=None, dim=DIM):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*dim, noise)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, dim, 4, 4])

    output = lib.ops.deconv2d.Deconv2D('Generator.2', dim, dim, 5, output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.3', dim, dim, 5, output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.4', dim, dim, 5, output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.5', dim, 3, 5, output)
    output = tf.tanh(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])

def nchw_to_nhwc(x):
    return tf.transpose(x, [0,2,3,1])
def nhwc_to_nchw(x):
    return tf.transpose(x, [0,3,1,2])
def BEGANGenerator(n_samples, noise=None, dim=DIM):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = lib.ops.linear.Linear('Generator.Input', 128, 8*8*dim, noise)
    output = tf.reshape(output, [-1, dim, 8, 8])

    repeat_num = int(np.log2(TARGET_SIZE/8)) + 1
    curr_shape = 8
    for idx in range(repeat_num):
        output = lib.ops.conv2d.Conv2D('Generator.{}a'.format(idx+2), dim, dim, 3, output, stride=1)
        output = LeakyReLU(output)
        output = lib.ops.conv2d.Conv2D('Generator.{}b'.format(idx+2), dim, dim, 3, output, stride=1)
        output = LeakyReLU(output)
        if idx < repeat_num - 1:
            output = nchw_to_nhwc(output)
            output = tf.image.resize_nearest_neighbor(output, (curr_shape*2, curr_shape*2))
            output = nhwc_to_nchw(output)
            curr_shape = curr_shape * 2
    print(curr_shape)

    output = lib.ops.conv2d.Conv2D('Generator.{}'.format(repeat_num+2), dim, 3, 3, output)
    output = tf.tanh(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])
def ResnetGenerator(n_samples, noise=None, dim=DIM):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*8*dim, noise)
    output = tf.reshape(output, [-1, 8*dim, 4, 4])

    for i in range(6):
        output = ResidualBlock('Generator.4x4_{}'.format(i), 8*dim, 8*dim, 3, output, resample=None)
    output = ResidualBlock('Generator.Up1', 8*dim, 4*dim, 3, output, resample='up')
    for i in range(6):
        output = ResidualBlock('Generator.8x8_{}'.format(i), 4*dim, 4*dim, 3, output, resample=None)
    output = ResidualBlock('Generator.Up2', 4*dim, 2*dim, 3, output, resample='up')
    for i in range(6):
        output = ResidualBlock('Generator.16x16_{}'.format(i), 2*dim, 2*dim, 3, output, resample=None)
    output = ResidualBlock('Generator.Up3', 2*dim, 1*dim, 3, output, resample='up')
    for i in range(6):
        output = ResidualBlock('Generator.32x32_{}'.format(i), 1*dim, 1*dim, 3, output, resample=None)
    output = ResidualBlock('Generator.Up4', 1*dim, dim/2, 3, output, resample='up')
    for i in range(5):
        output = ResidualBlock('Generator.64x64_{}'.format(i), dim/2, dim/2, 3, output, resample=None)

    output = lib.ops.conv2d.Conv2D('Generator.Out', dim/2, 3, 1, output, he_init=False)
    output = tf.tanh(output / 5.)

    return tf.reshape(output, [-1, OUTPUT_DIM])


def MultiplicativeDCGANGenerator(n_samples, noise=None, dim=DIM, bn=True):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*8*dim*2, noise)
    output = tf.reshape(output, [-1, 8*dim*2, 4, 4])
    if bn:
        output = Batchnorm('Generator.BN1', [0,2,3], output)
    output = pixcnn_gated_nonlinearity(output[:,::2], output[:,1::2])

    output = lib.ops.deconv2d.Deconv2D('Generator.2', 8*dim, 4*dim*2, 5, output)
    if bn:
        output = Batchnorm('Generator.BN2', [0,2,3], output)
    output = pixcnn_gated_nonlinearity(output[:,::2], output[:,1::2])

    output = lib.ops.deconv2d.Deconv2D('Generator.3', 4*dim, 2*dim*2, 5, output)
    if bn:
        output = Batchnorm('Generator.BN3', [0,2,3], output)
    output = pixcnn_gated_nonlinearity(output[:,::2], output[:,1::2])

    output = lib.ops.deconv2d.Deconv2D('Generator.4', 2*dim, dim*2, 5, output)
    if bn:
        output = Batchnorm('Generator.BN4', [0,2,3], output)
    output = pixcnn_gated_nonlinearity(output[:,::2], output[:,1::2])

    output = lib.ops.deconv2d.Deconv2D('Generator.5', dim, 3, 5, output)
    output = tf.tanh(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])

# ! Discriminators

def MultiplicativeDCGANDiscriminator(inputs, dim=DIM, bn=True):
    output = tf.reshape(inputs, [-1, 3, 64, 64])

    output = lib.ops.conv2d.Conv2D('Discriminator.1', 3, dim*2, 5, output, stride=2)
    output = pixcnn_gated_nonlinearity(output[:,::2], output[:,1::2])

    output = lib.ops.conv2d.Conv2D('Discriminator.2', dim, 2*dim*2, 5, output, stride=2)
    if bn:
        output = Batchnorm('Discriminator.BN2', [0,2,3], output)
    output = pixcnn_gated_nonlinearity(output[:,::2], output[:,1::2])

    output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*dim, 4*dim*2, 5, output, stride=2)
    if bn:
        output = Batchnorm('Discriminator.BN3', [0,2,3], output)
    output = pixcnn_gated_nonlinearity(output[:,::2], output[:,1::2])

    output = lib.ops.conv2d.Conv2D('Discriminator.4', 4*dim, 8*dim*2, 5, output, stride=2)
    if bn:
        output = Batchnorm('Discriminator.BN4', [0,2,3], output)
    output = pixcnn_gated_nonlinearity(output[:,::2], output[:,1::2])

    output = tf.reshape(output, [-1, 4*4*8*dim])
    output = lib.ops.linear.Linear('Discriminator.Output', 4*4*8*dim, 1, output)

    return tf.reshape(output, [-1])


def ResnetDiscriminator(inputs, dim=DIM):
    output = tf.reshape(inputs, [-1, 3, 64, 64])
    output = lib.ops.conv2d.Conv2D('Discriminator.In', 3, dim/2, 1, output, he_init=False)

    for i in range(5):
        output = ResidualBlock('Discriminator.64x64_{}'.format(i), dim/2, dim/2, 3, output, resample=None)
    output = ResidualBlock('Discriminator.Down1', dim/2, dim*1, 3, output, resample='down')
    for i in range(6):
        output = ResidualBlock('Discriminator.32x32_{}'.format(i), dim*1, dim*1, 3, output, resample=None)
    output = ResidualBlock('Discriminator.Down2', dim*1, dim*2, 3, output, resample='down')
    for i in range(6):
        output = ResidualBlock('Discriminator.16x16_{}'.format(i), dim*2, dim*2, 3, output, resample=None)
    output = ResidualBlock('Discriminator.Down3', dim*2, dim*4, 3, output, resample='down')
    for i in range(6):
        output = ResidualBlock('Discriminator.8x8_{}'.format(i), dim*4, dim*4, 3, output, resample=None)
    output = ResidualBlock('Discriminator.Down4', dim*4, dim*8, 3, output, resample='down')
    for i in range(6):
        output = ResidualBlock('Discriminator.4x4_{}'.format(i), dim*8, dim*8, 3, output, resample=None)

    output = tf.reshape(output, [-1, 4*4*8*dim])
    output = lib.ops.linear.Linear('Discriminator.Output', 4*4*8*dim, 1, output)

    return tf.reshape(output / 5., [-1])


def FCDiscriminator(inputs, FC_DIM=512, n_layers=3):
    output = LeakyReLULayer('Discriminator.Input', OUTPUT_DIM, FC_DIM, inputs)
    for i in range(n_layers):
        output = LeakyReLULayer('Discriminator.{}'.format(i), FC_DIM, FC_DIM, output)
    output = lib.ops.linear.Linear('Discriminator.Out', FC_DIM, 1, output)

    return tf.reshape(output, [-1])

def DCGANDiscriminator(inputs, dim=DIM, bn=True, nonlinearity=LeakyReLU):
    output = tf.reshape(inputs, [-1, 3, 64, 64])

    lib.ops.conv2d.set_weights_stdev(0.02)
    lib.ops.deconv2d.set_weights_stdev(0.02)
    lib.ops.linear.set_weights_stdev(0.02)

    output = lib.ops.conv2d.Conv2D('Discriminator.1', 3, dim, 5, output, stride=2)
    output = nonlinearity(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.2', dim, 2*dim, 5, output, stride=2)
    if bn:
        output = Batchnorm('Discriminator.BN2', [0,2,3], output)
    output = nonlinearity(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*dim, 4*dim, 5, output, stride=2)
    if bn:
        output = Batchnorm('Discriminator.BN3', [0,2,3], output)
    output = nonlinearity(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.4', 4*dim, 8*dim, 5, output, stride=2)
    if bn:
        output = Batchnorm('Discriminator.BN4', [0,2,3], output)
    output = nonlinearity(output)

    output = tf.reshape(output, [-1, 4*4*8*dim])
    output = lib.ops.linear.Linear('Discriminator.Output', 4*4*8*dim, 1, output)

    lib.ops.conv2d.unset_weights_stdev()
    lib.ops.deconv2d.unset_weights_stdev()
    lib.ops.linear.unset_weights_stdev()

    return tf.reshape(output, [-1])
	
def DCGANDiscriminator_128(inputs, dim=DIM, bn=True, nonlinearity=LeakyReLU):
    output = tf.reshape(inputs, [-1, 3, 128, 128])

    lib.ops.conv2d.set_weights_stdev(0.02)
    lib.ops.deconv2d.set_weights_stdev(0.02)
    lib.ops.linear.set_weights_stdev(0.02)

    output = lib.ops.conv2d.Conv2D('Discriminator.1', 3, dim, 5, output, stride=2)
    output = nonlinearity(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.2', dim, 2*dim, 5, output, stride=2)
    if bn:
        output = Batchnorm('Discriminator.BN2', [0,2,3], output)
    output = nonlinearity(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*dim, 4*dim, 5, output, stride=2)
    if bn:
        output = Batchnorm('Discriminator.BN3', [0,2,3], output)
    output = nonlinearity(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.4', 4*dim, 8*dim, 5, output, stride=2)
    if bn:
        output = Batchnorm('Discriminator.BN4', [0,2,3], output)
    output = nonlinearity(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.5', 8*dim, 16*dim, 5, output, stride=2)
    if bn:
        output = Batchnorm('Discriminator.BN5', [0,2,3], output)
    output = nonlinearity(output)

    output = tf.reshape(output, [-1, 4*4*16*dim])
    output = lib.ops.linear.Linear('Discriminator.Output', 4*4*16*dim, 1, output)

    lib.ops.conv2d.unset_weights_stdev()
    lib.ops.deconv2d.unset_weights_stdev()
    lib.ops.linear.unset_weights_stdev()

    return tf.reshape(output, [-1])


Generator, Discriminator = GeneratorAndDiscriminator()

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:

    all_real_data_conv = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 3, TARGET_SIZE, TARGET_SIZE])
    if tf.__version__.startswith('1.'):
        split_real_data_conv = tf.split(all_real_data_conv, len(DEVICES))
    else:
        split_real_data_conv = tf.split(0, len(DEVICES), all_real_data_conv)
    gen_costs, disc_costs = [],[]

    for device_index, (device, real_data_conv) in enumerate(zip(DEVICES, split_real_data_conv)):
        with tf.device(device):
            real_data = tf.reshape(2*((tf.cast(real_data_conv, tf.float32)/255.)-.5), [int(BATCH_SIZE/len(DEVICES)), OUTPUT_DIM])
            fake_data = Generator(int(BATCH_SIZE/len(DEVICES)))

            disc_real = Discriminator(real_data)
            disc_fake = Discriminator(fake_data)

            if MODE == 'wgan':
                gen_cost = -tf.reduce_mean(disc_fake)
                disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

            elif MODE == 'wgan-gp':
                gen_cost = -tf.reduce_mean(disc_fake)
                disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

                alpha = tf.random_uniform(
                    shape=[int(BATCH_SIZE/len(DEVICES)),1], 
                    minval=0.,
                    maxval=1.
                )
                differences = fake_data - real_data
                interpolates = real_data + (alpha*differences)
                gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
                slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
                gradient_penalty = tf.reduce_mean((slopes-1.)**2)
                disc_cost += LAMBDA*gradient_penalty

            elif MODE == 'dcgan':
                try: # tf pre-1.0 (bottom) vs 1.0 (top)
                    gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake,
                                                                                      labels=tf.ones_like(disc_fake)))
                    disc_cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake,
                                                                                        labels=tf.zeros_like(disc_fake)))
                    disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real,
                                                                                        labels=tf.ones_like(disc_real)))                    
                except Exception as e:
                    gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake, tf.ones_like(disc_fake)))
                    disc_cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake, tf.zeros_like(disc_fake)))
                    disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_real, tf.ones_like(disc_real)))                    
                disc_cost /= 2.

            elif MODE == 'lsgan':
                gen_cost = tf.reduce_mean((disc_fake - 1)**2)
                disc_cost = (tf.reduce_mean((disc_real - 1)**2) + tf.reduce_mean((disc_fake - 0)**2))/2.

            else:
                raise Exception()

            gen_costs.append(gen_cost)
            disc_costs.append(disc_cost)

    gen_cost = tf.add_n(gen_costs) / len(DEVICES)
    disc_cost = tf.add_n(disc_costs) / len(DEVICES)

    if MODE == 'wgan':
        gen_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(gen_cost,
                                             var_list=lib.params_with_name('Generator'), colocate_gradients_with_ops=True)
        disc_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(disc_cost,
                                             var_list=lib.params_with_name('Discriminator.'), colocate_gradients_with_ops=True)

        clip_ops = []
        for var in lib.params_with_name('Discriminator'):
            clip_bounds = [-.01, .01]
            clip_ops.append(tf.assign(var, tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])))
        clip_disc_weights = tf.group(*clip_ops)

    elif MODE == 'wgan-gp':
        gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost,
                                          var_list=lib.params_with_name('Generator'), colocate_gradients_with_ops=True)
        disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost,
                                           var_list=lib.params_with_name('Discriminator.'), colocate_gradients_with_ops=True)

    elif MODE == 'dcgan':
        gen_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(gen_cost,
                                          var_list=lib.params_with_name('Generator'), colocate_gradients_with_ops=True)
        disc_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(disc_cost,
                                           var_list=lib.params_with_name('Discriminator.'), colocate_gradients_with_ops=True)

    elif MODE == 'lsgan':
        gen_train_op = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(gen_cost,
                                             var_list=lib.params_with_name('Generator'), colocate_gradients_with_ops=True)
        disc_train_op = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(disc_cost,
                                              var_list=lib.params_with_name('Discriminator.'), colocate_gradients_with_ops=True)

    else:
        raise Exception()

    # For generating samples
    fixed_noise = tf.constant(np.random.normal(size=(BATCH_SIZE, 128)).astype('float32'))
    all_fixed_noise_samples = []
    for device_index, device in enumerate(DEVICES):
        n_samples = int(BATCH_SIZE / len(DEVICES))
        all_fixed_noise_samples.append(Generator(n_samples, noise=fixed_noise[device_index*n_samples:(device_index+1)*n_samples]))
    if tf.__version__.startswith('1.'):
        all_fixed_noise_samples = tf.concat(all_fixed_noise_samples, axis=0)
    else:
        all_fixed_noise_samples = tf.concat(0, all_fixed_noise_samples)
    def generate_image(iteration):
        samples = session.run(all_fixed_noise_samples)
        samples = ((samples+1.)*(255.99/2)).astype('int32')
        lib.save_images.save_images(samples.reshape((BATCH_SIZE, 3, TARGET_SIZE, TARGET_SIZE)), 'samples/{}/samples_{}.png'.format(EXP_NAME, iteration))

    saver = tf.train.Saver(tf.global_variables())

    # Dataset iterator
    #train_gen, dev_gen = lib.small_imagenet.load(BATCH_SIZE, data_dir=DATA_DIR)
    train_gen, dev_gen = lib.oneclass.load(DATA_DIR, CLASSNAME, image_size=TARGET_SIZE, batch_size= BATCH_SIZE, is_crop=IS_CROP)

    def inf_train_gen():
        while True:
            for (images,) in train_gen():
                yield images

    # Save a batch of ground-truth samples
    _x = next(inf_train_gen())
    _x_r = session.run(real_data, feed_dict={real_data_conv: _x})
    _x_r = ((_x_r+1.)*(255.99/2)).astype('int32')
    lib.save_images.save_images(_x_r.reshape((BATCH_SIZE, 3, TARGET_SIZE, TARGET_SIZE)), 'samples/{}/samples_groundtruth.png'.format(EXP_NAME))


    # Train loop
    if LOAD_ITER == 0:
        session.run(tf.initialize_all_variables())
    else:
        saver.restore(session, LOAD_PATH)
        print('loaded model from {}'.format(LOAD_PATH))
    gen = inf_train_gen()
    for iteration in range(LOAD_ITER, ITERS):

        start_time = time.time()

        # Train generator
        if iteration > 0:
            _ = session.run(gen_train_op)

        # Train critic
        if (MODE == 'dcgan') or (MODE == 'lsgan'):
            disc_iters = 1
        else:
            disc_iters = CRITIC_ITERS
        for i in range(disc_iters):
            _data = next(gen)
            _disc_cost, _ = session.run([disc_cost, disc_train_op], feed_dict={all_real_data_conv: _data})
            if MODE == 'wgan':
                _ = session.run([clip_disc_weights])

        lib.plot.plot('samples/{}/train disc cost'.format(EXP_NAME), _disc_cost)
        lib.plot.plot('samples/{}/time'.format(EXP_NAME), time.time() - start_time)

        if iteration % 200 == 199:
            t = time.time()
            dev_disc_costs = []
            for (images,) in dev_gen():
                _dev_disc_cost = session.run(disc_cost, feed_dict={all_real_data_conv: _data}) 
                dev_disc_costs.append(_dev_disc_cost)
            lib.plot.plot('samples/{}/dev disc cost'.format(EXP_NAME), np.mean(dev_disc_costs))

            generate_image(iteration)

        if (iteration < 5) or (iteration % 200 == 199):
            lib.plot.flush('samples/{}/'.format(EXP_NAME))

        if iteration % 10000 == 0:
            saver.save(session, 'models/{}/model.ckpt'.format(EXP_NAME), global_step=iteration)
		
        lib.plot.tick()
