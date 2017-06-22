#coding=utf-8
import os, sys, cv2, pickle
from multiprocessing import Pool
from functools import partial
from time import time
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from utils import *
from scipy import misc, ndimage, signal, sparse, io

from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Flatten,Activation,Lambda
from keras.layers.convolutional import Conv2D,MaxPooling2D,UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.regularizers import l2
from keras.optimizers import SGD, Adam
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint  

import argparse
parser = argparse.ArgumentParser(description='Train-Test-Deploy')
parser.add_argument('GPU', type=str, default="4",
                    help='Your GPU ID')
parser.add_argument('mode', type=str, default="train",
                    help='train-test, test or deploy')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=args.GPU
config = K.tf.ConfigProto(gpu_options=K.tf.GPUOptions(allow_growth=True))
sess = K.tf.Session(config=config)
K.set_session(sess)

batch_size = 2
use_multiprocessing = False

train_set = ['../datasets/CISL24218/',]
train_sample_rate = None
test_set = ['../datasets/NISTSD27/',]
deploy_set = ['../datasets/NISTSD27/images/','../datasets/CISL24218/', \
            '../datasets/FVC2002DB2A/','../datasets/NIST4/','../datasets/NIST14/']
pretrain = '../models/released_version/Model.model'
output_dir = '../output/'+datetime.now().strftime('%Y%m%d-%H%M%S')
logging = init_log(output_dir)
copy_file(sys.path[0]+'/'+sys.argv[0], output_dir+'/')

# image normalization
def img_normalization(img_input, m0=0.0, var0=1.0):
    m = K.mean(img_input, axis=[1,2,3], keepdims=True)
    var = K.var(img_input, axis=[1,2,3], keepdims=True)
    after = K.sqrt(var0*K.tf.square(img_input-m)/var)
    image_n = K.tf.where(K.tf.greater(img_input, m), m0+after, m0-after)
    return image_n

# atan2 function
def atan2(y_x):
    y, x = y_x[0], y_x[1]+K.epsilon()
    atan = K.tf.atan(y/x)
    angle = K.tf.where(K.tf.greater(x,0.0), atan, K.tf.zeros_like(x))
    angle = K.tf.where(K.tf.logical_and(K.tf.less(x,0.0),  K.tf.greater_equal(y,0.0)), atan+np.pi, angle)
    angle = K.tf.where(K.tf.logical_and(K.tf.less(x,0.0),  K.tf.less(y,0.0)), atan-np.pi, angle)
    return angle

# traditional orientation estimation
def orientation(image, stride=8, window=17):
    with K.tf.name_scope('orientation'):
        assert image.get_shape().as_list()[3] == 1, 'Images must be grayscale'
        strides = [1, stride, stride, 1]
        E = np.ones([window, window, 1, 1])
        sobelx = np.reshape(np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float), [3, 3, 1, 1])
        sobely = np.reshape(np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float), [3, 3, 1, 1])
        gaussian = np.reshape(gaussian2d((5, 5), 1), [5, 5, 1, 1])
        with K.tf.name_scope('sobel_gradient'):
            Ix = K.tf.nn.conv2d(image, sobelx, strides=[1,1,1,1], padding='SAME', name='sobel_x')
            Iy = K.tf.nn.conv2d(image, sobely, strides=[1,1,1,1], padding='SAME', name='sobel_y')
        with K.tf.name_scope('eltwise_1'):
            Ix2 = K.tf.multiply(Ix, Ix, name='IxIx')
            Iy2 = K.tf.multiply(Iy, Iy, name='IyIy')
            Ixy = K.tf.multiply(Ix, Iy, name='IxIy')
        with K.tf.name_scope('range_sum'):
            Gxx = K.tf.nn.conv2d(Ix2, E, strides=strides, padding='SAME', name='Gxx_sum')
            Gyy = K.tf.nn.conv2d(Iy2, E, strides=strides, padding='SAME', name='Gyy_sum')
            Gxy = K.tf.nn.conv2d(Ixy, E, strides=strides, padding='SAME', name='Gxy_sum')
        with K.tf.name_scope('eltwise_2'):
            Gxx_Gyy = K.tf.subtract(Gxx, Gyy, name='Gxx_Gyy')
            theta = atan2([2*Gxy, Gxx_Gyy]) + np.pi
        # two-dimensional low-pass filter: Gaussian filter here
        with K.tf.name_scope('gaussian_filter'):
            phi_x = K.tf.nn.conv2d(K.tf.cos(theta), gaussian, strides=[1,1,1,1], padding='SAME', name='gaussian_x')
            phi_y = K.tf.nn.conv2d(K.tf.sin(theta), gaussian, strides=[1,1,1,1], padding='SAME', name='gaussian_y')
            theta = atan2([phi_y, phi_x])/2
    return theta

def get_tra_ori():
    img_input=Input(shape=(None, None, 1))
    theta = Lambda(orientation)(img_input)
    model = Model(inputs=[img_input,], outputs=[theta,])
    return model
tra_ori_model = get_tra_ori()

def get_maximum_img_size_and_names(dataset, sample_rate=None):
    if sample_rate is None:
        sample_rate = [1]*len(dataset)
    img_name, folder_name, img_size = [], [], []
    for folder, rate in zip(dataset, sample_rate):
        _, img_name_t = get_files_in_folder(folder+'images/', '.bmp')
        img_name.extend(img_name_t.tolist()*rate)
        folder_name.extend([folder]*img_name_t.shape[0]*rate)
        img_size.append(np.array(misc.imread(folder+'images/'+img_name_t[0]+'.bmp', mode='L').shape))
    img_name = np.asarray(img_name)
    folder_name = np.asarray(folder_name)
    img_size = np.max(np.asarray(img_size), axis=0)
    # let img_size % 8 == 0
    img_size = np.array(np.ceil(img_size/8)*8,dtype=np.int32)
    return img_name, folder_name, img_size

def sub_load_data(data, img_size, aug): 
    img_name, dataset = data
    img = misc.imread(dataset+'images/'+img_name+'.bmp', mode='L')
    seg = misc.imread(dataset+'seg_labels/'+img_name+'.png', mode='L')
    try:
        ali = misc.imread(dataset+'ori_labels/'+img_name+'.bmp', mode='L')
    except:
        ali = np.zeros_like(img)
    mnt = np.array(mnt_reader(dataset+'mnt_labels/'+img_name+'.mnt'), dtype=float)
    if any(img.shape != img_size):
        # random pad mean values to reach required shape
        if np.random.rand()<aug:
            tra = np.int32(np.random.rand(2)*(np.array(img_size)-np.array(img.shape)))
        else:
            tra = np.int32(0.5*(np.array(img_size)-np.array(img.shape)))
        img_t = np.ones(img_size)*np.mean(img)
        seg_t = np.zeros(img_size)
        ali_t = np.ones(img_size)*np.mean(ali)
        img_t[tra[0]:tra[0]+img.shape[0],tra[1]:tra[1]+img.shape[1]] = img
        seg_t[tra[0]:tra[0]+img.shape[0],tra[1]:tra[1]+img.shape[1]] = seg
        ali_t[tra[0]:tra[0]+img.shape[0],tra[1]:tra[1]+img.shape[1]] = ali
        img = img_t
        seg = seg_t
        ali = ali_t
        mnt = mnt+np.array([tra[1],tra[0],0]) 
    if np.random.rand()<aug:
        # random rotation [0 - 360] & translation img_size / 4
        rot = np.random.rand() * 360
        tra = (np.random.rand(2)-0.5) / 2 * img_size 
        img = ndimage.rotate(img, rot, reshape=False, mode='reflect')
        img = ndimage.shift(img, tra, mode='reflect')
        seg = ndimage.rotate(seg, rot, reshape=False, mode='constant')
        seg = ndimage.shift(seg, tra, mode='constant')
        ali = ndimage.rotate(ali, rot, reshape=False, mode='reflect')
        ali = ndimage.shift(ali, tra, mode='reflect') 
        mnt_r = point_rot(mnt[:, :2], rot/180*np.pi, img.shape, img.shape)  
        mnt = np.column_stack((mnt_r+tra[[1, 0]], mnt[:, 2]-rot/180*np.pi))
    # only keep mnt that stay in pic & not on border
    mnt = mnt[(8<=mnt[:,0])*(mnt[:,0]<img_size[1]-8)*(8<=mnt[:, 1])*(mnt[:,1]<img_size[0]-8), :]
    return img, seg, ali, mnt   

def load_data(dataset, tra_ori_model, rand=False, aug=0.0, batch_size=1, sample_rate=None):
    if type(dataset[0]) == str:
        img_name, folder_name, img_size = get_maximum_img_size_and_names(dataset, sample_rate)
    else:
        img_name, folder_name, img_size = dataset
    if rand:
        rand_idx = np.arange(len(img_name))
        np.random.shuffle(rand_idx)
        img_name = img_name[rand_idx]
        folder_name = folder_name[rand_idx]
    if batch_size > 1 and use_multiprocessing==True:
        p = Pool(batch_size)        
    p_sub_load_data = partial(sub_load_data, img_size=img_size, aug=aug)
    for i in xrange(0,len(img_name), batch_size):
        have_alignment = np.ones([batch_size, 1, 1, 1])
        image = np.zeros((batch_size, img_size[0], img_size[1], 1))
        segment = np.zeros((batch_size, img_size[0], img_size[1], 1))
        alignment = np.zeros((batch_size, img_size[0], img_size[1], 1))
        minutiae_w = np.zeros((batch_size, img_size[0]/8, img_size[1]/8, 1))-1
        minutiae_h = np.zeros((batch_size, img_size[0]/8, img_size[1]/8, 1))-1
        minutiae_o = np.zeros((batch_size, img_size[0]/8, img_size[1]/8, 1))-1
        batch_name = [img_name[(i+j)%len(img_name)] for j in xrange(batch_size)]
        batch_f_name = [folder_name[(i+j)%len(img_name)] for j in xrange(batch_size)]
        if batch_size > 1 and use_multiprocessing==True:    
            results = p.map(p_sub_load_data, zip(batch_name, batch_f_name))
        else:
            results = map(p_sub_load_data, zip(batch_name, batch_f_name))
        for j in xrange(batch_size):
            img, seg, ali, mnt = results[j]
            if np.sum(ali) == 0:
                have_alignment[j, 0, 0, 0] = 0
            image[j, :, :, 0] = img / 255.0
            segment[j, :, :, 0] = seg / 255.0
            alignment[j, :, :, 0] = ali / 255.0
            minutiae_w[j, (mnt[:, 1]/8).astype(int), (mnt[:, 0]/8).astype(int), 0] = mnt[:, 0] % 8
            minutiae_h[j, (mnt[:, 1]/8).astype(int), (mnt[:, 0]/8).astype(int), 0] = mnt[:, 1] % 8
            minutiae_o[j, (mnt[:, 1]/8).astype(int), (mnt[:, 0]/8).astype(int), 0] = mnt[:, 2]
        # get seg
        label_seg = segment[:, ::8, ::8, :]
        label_seg[label_seg>0] = 1
        label_seg[label_seg<=0] = 0
        minutiae_seg = (minutiae_o!=-1).astype(float)
        # get ori & mnt
        orientation = tra_ori_model.predict(alignment)        
        orientation = orientation/np.pi*180+90
        orientation[orientation>=180.0] = 0.0 # orientation [0, 180)
        minutiae_o = minutiae_o/np.pi*180+90 # [90, 450)
        minutiae_o[minutiae_o>360] = minutiae_o[minutiae_o>360]-360 # to current coordinate system [0, 360)
        minutiae_ori_o = np.copy(minutiae_o) # copy one
        minutiae_ori_o[minutiae_ori_o>=180] = minutiae_ori_o[minutiae_ori_o>=180]-180 # for strong ori label [0,180)      
        # ori 2 gaussian
        gaussian_pdf = signal.gaussian(361, 3)
        y = np.reshape(np.arange(1, 180, 2), [1,1,1,-1])
        delta = np.array(np.abs(orientation - y), dtype=int)
        delta = np.minimum(delta, 180-delta)+180
        label_ori = gaussian_pdf[delta]
        # ori_o 2 gaussian
        delta = np.array(np.abs(minutiae_ori_o - y), dtype=int)
        delta = np.minimum(delta, 180-delta)+180
        label_ori_o = gaussian_pdf[delta] 
        # mnt_o 2 gaussian
        y = np.reshape(np.arange(1, 360, 2), [1,1,1,-1])
        delta = np.array(np.abs(minutiae_o - y), dtype=int)  
        delta = np.minimum(delta, 360-delta)+180
        label_mnt_o = gaussian_pdf[delta]         
        # w 2 gaussian
        gaussian_pdf = signal.gaussian(17, 2)
        y = np.reshape(np.arange(0, 8), [1,1,1,-1])
        delta = (minutiae_w-y+8).astype(int)
        label_mnt_w = gaussian_pdf[delta]
        # h 2 gaussian
        delta = (minutiae_h-y+8).astype(int)
        label_mnt_h = gaussian_pdf[delta]
        # mnt cls label -1:neg, 0:no care, 1:pos
        label_mnt_s = np.copy(minutiae_seg)
        label_mnt_s[label_mnt_s==0] = -1 # neg to -1
        label_mnt_s = (label_mnt_s+ndimage.maximum_filter(label_mnt_s, size=(1,3,3,1)))/2 # around 3*3 pos -> 0
        # apply segmentation
        label_ori = label_ori * label_seg * have_alignment
        label_ori_o = label_ori_o * minutiae_seg
        label_mnt_o = label_mnt_o * minutiae_seg
        label_mnt_w = label_mnt_w * minutiae_seg
        label_mnt_h = label_mnt_h * minutiae_seg
        yield image, label_ori, label_ori_o, label_seg, label_mnt_w, label_mnt_h, label_mnt_o, label_mnt_s, batch_name
    if batch_size > 1 and use_multiprocessing==True:
        p.close()
        p.join()
    return

def merge_mul(x):
    return reduce(lambda x,y:x*y, x)
def merge_sum(x):
    return reduce(lambda x,y:x+y, x)
def reduce_sum(x):
    return K.sum(x,axis=-1,keepdims=True) 
def merge_concat(x):
    return K.tf.concat(x,3)
def select_max(x):
    x = x / (K.max(x, axis=-1, keepdims=True)+K.epsilon())
    x = K.tf.where(K.tf.greater(x, 0.999), x, K.tf.zeros_like(x)) # select the biggest one
    x = x / (K.sum(x, axis=-1, keepdims=True)+K.epsilon()) # prevent two or more ori is selected
    return x  
def conv_bn(bottom, w_size, name, strides=(1,1), dilation_rate=(1,1)):
    top = Conv2D(w_size[0], (w_size[1],w_size[2]),
        kernel_regularizer=l2(5e-5),
        padding='same', 
        strides=strides,
        dilation_rate=dilation_rate,
        name='conv-'+name)(bottom)
    top = BatchNormalization(name='bn-'+name)(top)
    return top
def conv_bn_prelu(bottom, w_size, name, strides=(1,1), dilation_rate=(1,1)):
    if dilation_rate == (1,1):
        conv_type = 'conv'
    else:
        conv_type = 'atrousconv'
    top = Conv2D(w_size[0], (w_size[1],w_size[2]),
        kernel_regularizer=l2(5e-5),
        padding='same', 
        strides=strides,
        dilation_rate=dilation_rate,
        name=conv_type+name)(bottom)
    top = BatchNormalization(name='bn-'+name)(top)
    top=PReLU(alpha_initializer='zero', shared_axes=[1,2], name='prelu-'+name)(top)
    return top
def get_main_net(input_shape=(512,512,1), weights_path=None):
    img_input=Input(input_shape)
    bn_img=Lambda(img_normalization, name='img_norm')(img_input)
    # feature extraction VGG
    conv=conv_bn_prelu(bn_img, (64,3,3), '1_1') 
    conv=conv_bn_prelu(conv, (64,3,3), '1_2')
    conv=MaxPooling2D(pool_size=(2,2),strides=(2,2))(conv)

    conv=conv_bn_prelu(conv, (128,3,3), '2_1') 
    conv=conv_bn_prelu(conv, (128,3,3), '2_2') 
    conv=MaxPooling2D(pool_size=(2,2),strides=(2,2))(conv)

    conv=conv_bn_prelu(conv, (256,3,3), '3_1') 
    conv=conv_bn_prelu(conv, (256,3,3), '3_2') 
    conv=conv_bn_prelu(conv, (256,3,3), '3_3')   
    conv=MaxPooling2D(pool_size=(2,2),strides=(2,2))(conv)

    # multi-scale ASPP
    scale_1=conv_bn_prelu(conv, (256,3,3), '4_1', dilation_rate=(1,1))
    ori_1=conv_bn_prelu(scale_1, (128,1,1), 'ori_1_1')
    ori_1=Conv2D(90, (1,1), padding='same', name='ori_1_2')(ori_1)
    seg_1=conv_bn_prelu(scale_1, (128,1,1), 'seg_1_1')
    seg_1=Conv2D(1, (1,1), padding='same', name='seg_1_2')(seg_1)

    scale_2=conv_bn_prelu(conv, (256,3,3), '4_2', dilation_rate=(4,4))
    ori_2=conv_bn_prelu(scale_2, (128,1,1), 'ori_2_1')
    ori_2=Conv2D(90, (1,1), padding='same', name='ori_2_2')(ori_2)    
    seg_2=conv_bn_prelu(scale_2, (128,1,1), 'seg_2_1')
    seg_2=Conv2D(1, (1,1), padding='same', name='seg_2_2')(seg_2)

    scale_3=conv_bn_prelu(conv, (256,3,3), '4_3', dilation_rate=(8,8))
    ori_3=conv_bn_prelu(scale_3, (128,1,1), 'ori_3_1')
    ori_3=Conv2D(90, (1,1), padding='same', name='ori_3_2')(ori_3)  
    seg_3=conv_bn_prelu(scale_3, (128,1,1), 'seg_3_1')
    seg_3=Conv2D(1, (1,1), padding='same', name='seg_3_2')(seg_3)

    # sum fusion for ori
    ori_out=Lambda(merge_sum)([ori_1, ori_2, ori_3]) 
    ori_out_1=Activation('sigmoid', name='ori_out_1')(ori_out)
    ori_out_2=Activation('sigmoid', name='ori_out_2')(ori_out)

    # sum fusion for segmentation
    seg_out=Lambda(merge_sum)([seg_1, seg_2, seg_3])
    seg_out=Activation('sigmoid', name='seg_out')(seg_out)
    # ----------------------------------------------------------------------------
    # enhance part
    filters_cos, filters_sin = gabor_bank(stride=2, Lambda=8)
    filter_img_real = Conv2D(filters_cos.shape[3],(filters_cos.shape[0],filters_cos.shape[1]),
        weights=[filters_cos, np.zeros([filters_cos.shape[3]])], padding='same',
        name='enh_img_real_1')(img_input)
    filter_img_imag = Conv2D(filters_sin.shape[3],(filters_sin.shape[0],filters_sin.shape[1]),
        weights=[filters_sin, np.zeros([filters_sin.shape[3]])], padding='same',
        name='enh_img_imag_1')(img_input)
    ori_peak = Lambda(ori_highest_peak)(ori_out_1)
    ori_peak = Lambda(select_max)(ori_peak) # select max ori and set it to 1
    upsample_ori = UpSampling2D(size=(8,8))(ori_peak)
    seg_round = Activation('softsign')(seg_out)      
    upsample_seg = UpSampling2D(size=(8,8))(seg_round)
    mul_mask_real = Lambda(merge_mul)([filter_img_real, upsample_ori])
    enh_img_real = Lambda(reduce_sum, name='enh_img_real_2')(mul_mask_real)
    mul_mask_imag = Lambda(merge_mul)([filter_img_imag, upsample_ori])
    enh_img_imag = Lambda(reduce_sum, name='enh_img_imag_2')(mul_mask_imag)
    enh_img = Lambda(atan2, name='phase_img')([enh_img_imag, enh_img_real])
    enh_seg_img = Lambda(merge_concat, name='phase_seg_img')([enh_img, upsample_seg])
    # ----------------------------------------------------------------------------
    # mnt part
    mnt_conv=conv_bn_prelu(enh_seg_img, (64,9,9), 'mnt_1_1') 
    mnt_conv=MaxPooling2D(pool_size=(2,2),strides=(2,2))(mnt_conv)

    mnt_conv=conv_bn_prelu(mnt_conv, (128,5,5), 'mnt_2_1') 
    mnt_conv=MaxPooling2D(pool_size=(2,2),strides=(2,2))(mnt_conv)

    mnt_conv=conv_bn_prelu(mnt_conv, (256,3,3), 'mnt_3_1')  
    mnt_conv=MaxPooling2D(pool_size=(2,2),strides=(2,2))(mnt_conv)    

    mnt_o_1=Lambda(merge_concat)([mnt_conv, ori_out_1])
    mnt_o_2=conv_bn_prelu(mnt_o_1, (256,1,1), 'mnt_o_1_1')
    mnt_o_3=Conv2D(180, (1,1), padding='same', name='mnt_o_1_2')(mnt_o_2)
    mnt_o_out=Activation('sigmoid', name='mnt_o_out')(mnt_o_3)

    mnt_w_1=conv_bn_prelu(mnt_conv, (256,1,1), 'mnt_w_1_1')
    mnt_w_2=Conv2D(8, (1,1), padding='same', name='mnt_w_1_2')(mnt_w_1)
    mnt_w_out=Activation('sigmoid', name='mnt_w_out')(mnt_w_2)

    mnt_h_1=conv_bn_prelu(mnt_conv, (256,1,1), 'mnt_h_1_1')
    mnt_h_2=Conv2D(8, (1,1), padding='same', name='mnt_h_1_2')(mnt_h_1)
    mnt_h_out=Activation('sigmoid', name='mnt_h_out')(mnt_h_2) 

    mnt_s_1=conv_bn_prelu(mnt_conv, (256,1,1), 'mnt_s_1_1')
    mnt_s_2=Conv2D(1, (1,1), padding='same', name='mnt_s_1_2')(mnt_s_1)
    mnt_s_out=Activation('sigmoid', name='mnt_s_out')(mnt_s_2)

    if args.mode == 'deploy':
        model = Model(inputs=[img_input,], outputs=[enh_img_real, ori_out_1, ori_out_2, seg_out, mnt_o_out, mnt_w_out, mnt_h_out, mnt_s_out])
    else:
        model = Model(inputs=[img_input,], outputs=[ori_out_1, ori_out_2, seg_out, mnt_o_out, mnt_w_out, mnt_h_out, mnt_s_out])     
    if weights_path:
        model.load_weights(weights_path, by_name=True)
    return model   

kernal2angle = np.reshape(np.arange(1, 180, 2, dtype=float), [1,1,1,90])/90.*np.pi #2angle = angle*2
sin2angle, cos2angle = np.sin(kernal2angle), np.cos(kernal2angle)
def ori2angle(ori):
    sin2angle_ori = K.sum(ori*sin2angle, -1, keepdims=True)
    cos2angle_ori = K.sum(ori*cos2angle, -1, keepdims=True)
    modulus_ori = K.sqrt(K.square(sin2angle_ori)+K.square(cos2angle_ori))
    return sin2angle_ori, cos2angle_ori, modulus_ori

def ori_loss(y_true, y_pred, lamb=1.):
    # clip
    y_pred = K.tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
    # get ROI
    label_seg = K.sum(y_true, axis=-1, keepdims=True)
    label_seg = K.tf.cast(K.tf.greater(label_seg, 0), K.tf.float32) 
    # weighted cross entropy loss
    lamb_pos, lamb_neg = 1., 1. 
    logloss = lamb_pos*y_true*K.log(y_pred)+lamb_neg*(1-y_true)*K.log(1-y_pred)
    logloss = logloss*label_seg # apply ROI
    logloss = -K.sum(logloss) / (K.sum(label_seg) + K.epsilon())
    # coherence loss, nearby ori should be as near as possible
    mean_kernal = np.reshape(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32)/8, [3, 3, 1, 1])    
    sin2angle_ori, cos2angle_ori, modulus_ori = ori2angle(y_pred)
    sin2angle = K.conv2d(sin2angle_ori, mean_kernal, padding='same')
    cos2angle = K.conv2d(cos2angle_ori, mean_kernal, padding='same')
    modulus = K.conv2d(modulus_ori, mean_kernal, padding='same')
    coherence = K.sqrt(K.square(sin2angle) + K.square(cos2angle)) / (modulus + K.epsilon())
    coherenceloss = K.sum(label_seg) / (K.sum(coherence*label_seg) + K.epsilon()) - 1
    loss = logloss + lamb*coherenceloss
    return loss

def ori_o_loss(y_true, y_pred):
    # clip
    y_pred = K.tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
    # get ROI
    label_seg = K.sum(y_true, axis=-1, keepdims=True)
    label_seg = K.tf.cast(K.tf.greater(label_seg, 0), K.tf.float32) 
    # weighted cross entropy loss
    lamb_pos, lamb_neg= 1., 1. 
    logloss = lamb_pos*y_true*K.log(y_pred)+lamb_neg*(1-y_true)*K.log(1-y_pred)
    logloss = logloss*label_seg # apply ROI
    logloss = -K.sum(logloss) / (K.sum(label_seg) + K.epsilon())
    return logloss

def seg_loss(y_true, y_pred, lamb=1.):
    # clip
    y_pred = K.tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
    # weighted cross entropy loss
    total_elements = K.sum(K.tf.ones_like(y_true))
    label_pos = K.tf.cast(K.tf.greater(y_true, 0.0), K.tf.float32)   
    lamb_pos = 0.5 * total_elements / K.sum(label_pos)
    lamb_neg = 1 / (2 - 1/lamb_pos)
    logloss = lamb_pos*y_true*K.log(y_pred)+lamb_neg*(1-y_true)*K.log(1-y_pred)
    logloss = -K.mean(K.sum(logloss, axis=-1))
    # smooth loss
    smooth_kernal = np.reshape(np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)/8, [3, 3, 1, 1])
    smoothloss = K.mean(K.abs(K.conv2d(y_pred, smooth_kernal)))
    loss = logloss + lamb*smoothloss
    return loss

def mnt_s_loss(y_true, y_pred):
    # clip
    y_pred = K.tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
    # get ROI
    label_seg = K.tf.cast(K.tf.not_equal(y_true, 0.0), K.tf.float32) 
    y_true = K.tf.where(K.tf.less(y_true,0.0), K.tf.zeros_like(y_true), y_true) # set -1 -> 0
    # weighted cross entropy loss       
    total_elements = K.sum(label_seg) + K.epsilon()  
    lamb_pos, lamb_neg = 10., .5
    logloss = lamb_pos*y_true*K.log(y_pred)+lamb_neg*(1-y_true)*K.log(1-y_pred)
    # apply ROI
    logloss = logloss*label_seg
    logloss = -K.sum(logloss) / total_elements
    return logloss    

# find highest peak using gaussian
def ori_highest_peak(y_pred, length=180):
    glabel = gausslabel(length=length,stride=2).astype(np.float32)
    ori_gau = K.conv2d(y_pred,glabel,padding='same')
    return ori_gau

def ori_acc_delta_k(y_true, y_pred, k=10, max_delta=180):
    # get ROI
    label_seg = K.sum(y_true, axis=-1)
    label_seg = K.tf.cast(K.tf.greater(label_seg, 0), K.tf.float32) 
    # get pred angle    
    angle = K.cast(K.argmax(ori_highest_peak(y_pred, max_delta), axis=-1), dtype=K.tf.float32)*2.0+1.0
    # get gt angle
    angle_t = K.cast(K.argmax(y_true, axis=-1), dtype=K.tf.float32)*2.0+1.0
    # get delta
    angle_delta = K.abs(angle_t - angle)
    acc = K.tf.less_equal(K.minimum(angle_delta, max_delta-angle_delta), k)
    acc = K.cast(acc, dtype=K.tf.float32)
    # apply ROI
    acc = acc*label_seg
    acc = K.sum(acc) / (K.sum(label_seg)+K.epsilon())
    return acc
def ori_acc_delta_10(y_true, y_pred):
    return ori_acc_delta_k(y_true, y_pred, 10)
def ori_acc_delta_20(y_true, y_pred):
    return ori_acc_delta_k(y_true, y_pred, 20)
def mnt_acc_delta_10(y_true, y_pred):
    return ori_acc_delta_k(y_true, y_pred, 10, 360)
def mnt_acc_delta_20(y_true, y_pred):
    return ori_acc_delta_k(y_true, y_pred, 20, 360)    

def seg_acc_pos(y_true, y_pred):
    y_true = K.tf.where(K.tf.less(y_true,0.0), K.tf.zeros_like(y_true), y_true)
    acc = K.cast(K.equal(y_true, K.round(y_pred)), dtype=K.tf.float32)
    acc = K.sum(acc * y_true) / (K.sum(y_true)+K.epsilon())
    return acc    
def seg_acc_neg(y_true, y_pred):
    y_true = K.tf.where(K.tf.less(y_true,0.0), K.tf.zeros_like(y_true), y_true)
    acc = K.cast(K.equal(y_true, K.round(y_pred)), dtype=K.tf.float32)
    acc = K.sum(acc * (1-y_true)) / (K.sum(1-y_true)+K.epsilon())
    return acc
def seg_acc_all(y_true, y_pred):
    y_true = K.tf.where(K.tf.less(y_true,0.0), K.tf.zeros_like(y_true), y_true)
    return K.mean(K.equal(y_true, K.round(y_pred)))  

def mnt_mean_delta(y_true, y_pred):
    # get ROI
    label_seg = K.sum(y_true, axis=-1)
    label_seg = K.tf.cast(K.tf.greater(label_seg, 0), K.tf.float32) 
    # get pred pos    
    pos = K.cast(K.argmax(y_pred, axis=-1), dtype=K.tf.float32)
    # get gt pos
    pos_t = K.cast(K.argmax(y_true, axis=-1), dtype=K.tf.float32)
    # get delta
    pos_delta = K.abs(pos_t - pos)
    # apply ROI
    pos_delta = pos_delta*label_seg
    mean_delta = K.sum(pos_delta) / (K.sum(label_seg)+K.epsilon())
    return mean_delta

def train(input_shape=(512,512,1)):
    img_name, folder_name, img_size = get_maximum_img_size_and_names(train_set, train_sample_rate)  
    main_net_model = get_main_net((img_size[0],img_size[1],1), pretrain)
    plot_model(main_net_model, to_file=output_dir+'/model.png',show_shapes=True)
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)    
    main_net_model.compile(optimizer=adam, 
        loss={'ori_out_1':ori_loss, 'ori_out_2':ori_o_loss, 'seg_out':seg_loss, 
                'mnt_o_out':ori_o_loss, 'mnt_w_out':ori_o_loss, 'mnt_h_out':ori_o_loss, 'mnt_s_out':mnt_s_loss}, 
        loss_weights={'ori_out_1':.1, 'ori_out_2':.1, 'seg_out':10., 
                'mnt_w_out':.5, 'mnt_h_out':.5, 'mnt_o_out':.5,'mnt_s_out':200.},
        metrics={'ori_out_1':[ori_acc_delta_10,],
                 'ori_out_2':[ori_acc_delta_10,],
                 'seg_out':[seg_acc_pos, seg_acc_neg, seg_acc_all],
                 'mnt_o_out':[mnt_acc_delta_10,],
                 'mnt_w_out':[mnt_mean_delta,],
                 'mnt_h_out':[mnt_mean_delta,],
                 'mnt_s_out':[seg_acc_pos, seg_acc_neg, seg_acc_all]})
    for epoch in range(100):
        for i, train in enumerate(load_data((img_name, folder_name, img_size), tra_ori_model, rand=True, aug=0.7, batch_size=batch_size)):
            loss = main_net_model.train_on_batch(train[0], 
                {'ori_out_1':train[1], 'ori_out_2':train[2], 'seg_out':train[3],
                'mnt_w_out':train[4], 'mnt_h_out':train[5], 'mnt_o_out':train[6], 'mnt_s_out':train[7]})  
            if i%(20/batch_size) == 0:
                logging.info("epoch=%d, step=%d", epoch, i)
                logging.info("%s", " ".join(["%s:%.4f\n"%(x) for x in zip(main_net_model.metrics_names, loss)]))
            if i%(10000/batch_size) == (10000/batch_size)-1:
                # test every 10000 pics
                outdir = "%s/%03d_%05d/"%(output_dir, epoch, i)
                re_mkdir(outdir)
                savedir = "%s%s"%(outdir, str(epoch)+'_'+str(i))
                main_net_model.save_weights(savedir, True)
                for folder in test_set:
                    test([folder,], savedir, outdir, test_num=10, draw=False)
    return

# currently can only produce one each time
def label2mnt(mnt_s_out, mnt_w_out, mnt_h_out, mnt_o_out, thresh=0.5):
    mnt_s_out = np.squeeze(mnt_s_out)
    mnt_w_out = np.squeeze(mnt_w_out)
    mnt_h_out = np.squeeze(mnt_h_out)
    mnt_o_out = np.squeeze(mnt_o_out)
    assert len(mnt_s_out.shape)==2 and len(mnt_w_out.shape)==3 and len(mnt_h_out.shape)==3 and len(mnt_o_out.shape)==3 
    # get cls results
    mnt_sparse = sparse.coo_matrix(mnt_s_out>thresh)
    mnt_list = np.array(zip(mnt_sparse.row, mnt_sparse.col), dtype=np.int32)
    if mnt_list.shape[0] == 0:
        return np.zeros((0, 4))
    # get regression results
    mnt_w_out = np.argmax(mnt_w_out, axis=-1)
    mnt_h_out = np.argmax(mnt_h_out, axis=-1)
    mnt_o_out = np.argmax(mnt_o_out, axis=-1) # TODO: use ori_highest_peak(np version)
    # get final mnt
    mnt_final = np.zeros((len(mnt_list), 4))
    mnt_final[:, 0] = mnt_sparse.col*8 + mnt_w_out[mnt_list[:,0], mnt_list[:,1]]
    mnt_final[:, 1] = mnt_sparse.row*8 + mnt_h_out[mnt_list[:,0], mnt_list[:,1]]
    mnt_final[:, 2] = (mnt_o_out[mnt_list[:,0], mnt_list[:,1]]*2-89.)/180*np.pi
    mnt_final[mnt_final[:, 2]<0.0, 2] = mnt_final[mnt_final[:, 2]<0.0, 2]+2*np.pi
    mnt_final[:, 3] = mnt_s_out[mnt_list[:,0], mnt_list[:, 1]]
    return mnt_final
def test(test_set, model, outdir, test_num=10, draw=True):
    logging.info("Testing %s:"%(test_set))
    img_name, folder_name, img_size = get_maximum_img_size_and_names(test_set)  
    main_net_model = get_main_net((img_size[0],img_size[1],1), model)
    nonsense = SGD(lr=0.0, momentum=0.0, decay=0.0, nesterov=False)    
    main_net_model.compile(optimizer=nonsense,
        loss={'ori_out_1':ori_loss, 'ori_out_2':ori_o_loss, 'seg_out':seg_loss, 
                'mnt_o_out':ori_o_loss, 'mnt_w_out':ori_o_loss, 'mnt_h_out':ori_o_loss, 'mnt_s_out':mnt_s_loss}, 
        loss_weights={'ori_out_1':.1, 'ori_out_2':.1, 'seg_out':10., 
                'mnt_w_out':.5, 'mnt_h_out':.5, 'mnt_o_out':.5,'mnt_s_out':200.},        
        metrics={'ori_out_1':[ori_acc_delta_10,ori_acc_delta_20],
                 'ori_out_2':[ori_acc_delta_10,ori_acc_delta_20],
                 'seg_out':[seg_acc_pos, seg_acc_neg, seg_acc_all],
                 'mnt_o_out':[mnt_acc_delta_10,mnt_acc_delta_20],
                 'mnt_w_out':[mnt_mean_delta,],
                 'mnt_h_out':[mnt_mean_delta,],
                 'mnt_s_out':[seg_acc_pos, seg_acc_neg, seg_acc_all]})
    ave_loss, ave_prf_nms = [], []
    for j, test in enumerate(load_data((img_name, folder_name, img_size), tra_ori_model, rand=False, aug=0.0, batch_size=1)):      
        if j < test_num:
            logging.info("%d / %d: %s"%(j+1, len(img_name), img_name[j]))    
            ori_out_1, ori_out_2, seg_out, mnt_o_out, mnt_w_out, mnt_h_out, mnt_s_out  = main_net_model.predict(test[0])
            metrics = main_net_model.train_on_batch(test[0], 
                {'ori_out_1':test[1], 'ori_out_2':test[2], 'seg_out':test[3],
                'mnt_w_out':test[4], 'mnt_h_out':test[5], 'mnt_o_out':test[6], 'mnt_s_out':test[7]})  
            ave_loss.append(metrics)
            logging.info("%s", " ".join(["%s:%.4f\n"%(x) for x in zip(main_net_model.metrics_names, metrics)]))
            mnt_gt = label2mnt(test[7], test[4], test[5], test[6])
            mnt_s_out = mnt_s_out * test[3]
            mnt = label2mnt(mnt_s_out, mnt_w_out, mnt_h_out, mnt_o_out, thresh=0.5)
            mnt_nms = nms(mnt)
            p, r, f, l, o = mnt_P_R_F(mnt_gt, mnt_nms)
            logging.info("After_nms:\nprecision: %f\nrecall: %f\nf1-measure: %f\nlocation_dis: %f\norientation_delta:%f\n----------------\n"%(
                p, r, f, l, o))
            ave_prf_nms.append([p, r, f, l, o])            
            if draw:                         
                angval = sess.run(ori_highest_peak(ori_out_1))                           
                angval = (np.argmax(angval, axis=-1)*2-90)/180.*np.pi
                draw_ori_on_img(test[0], angval, seg_out, "%s%s_ori.png"%(outdir, test[8][0]))
                draw_minutiae(test[0], mnt_nms[:,:3], "%s%s_mnt.png"%(outdir, test[8][0]))
                draw_minutiae(test[0], mnt_gt[:,:3], "%s%s_mnt_gt.png"%(outdir, test[8][0]))
        else:
            break
    logging.info("Average testing results:")
    ave_loss = np.mean(np.array(ave_loss), 0)
    ave_prf_nms = np.mean(np.array(ave_prf_nms), 0)
    logging.info("\n%s\n", " ".join(["%s:%.4f\n"%(x) for x in zip(main_net_model.metrics_names, ave_loss)]))
    logging.info("After_nms:\nprecision: %f\nrecall: %f\nf1-measure: %f\nlocation_dis: %f\norientation_delta:%f\n----------------\n"%(
                    ave_prf_nms[0],ave_prf_nms[1],ave_prf_nms[2],ave_prf_nms[3],ave_prf_nms[4]))     
    return

def deploy(deploy_set, set_name=None):
    if set_name is None:
        set_name = deploy_set.split('/')[-2]
    mkdir(output_dir+'/'+set_name+'/')
    logging.info("Predicting %s:"%(set_name)) 
    _, img_name = get_files_in_folder(deploy_set, '.bmp')
    if len(img_name) == 0:
        deploy_set = deploy_set+'images/'
        _, img_name = get_files_in_folder(deploy_set, '.bmp')
    img_size = misc.imread(deploy_set+img_name[0]+'.bmp', mode='L').shape
    img_size = np.array(img_size, dtype=np.int32)/8*8      
    main_net_model = get_main_net((img_size[0],img_size[1],1), pretrain)
    _, img_name = get_files_in_folder(deploy_set, '.bmp')
    time_c = []
    for i in xrange(0,len(img_name)):
        logging.info("%s %d / %d: %s"%(set_name, i+1, len(img_name), img_name[i]))
        time_start = time()    
        image = misc.imread(deploy_set+img_name[i]+'.bmp', mode='L') / 255.0
        image = image[:img_size[0],:img_size[1]]      
        image = np.reshape(image,[1, image.shape[0], image.shape[1], 1])
        enhance_img, ori_out_1, ori_out_2, seg_out, mnt_o_out, mnt_w_out, mnt_h_out, mnt_s_out = main_net_model.predict(image) 
        time_afterconv = time()
        round_seg = np.round(np.squeeze(seg_out))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
        seg_out = cv2.morphologyEx(round_seg, cv2.MORPH_OPEN, kernel)
        mnt = label2mnt(np.squeeze(mnt_s_out)*np.round(np.squeeze(seg_out)), mnt_w_out, mnt_h_out, mnt_o_out, thresh=0.5)
        mnt_nms = nms(mnt)
        ori = sess.run(ori_highest_peak(ori_out_1))                           
        ori = (np.argmax(ori, axis=-1)*2-90)/180.*np.pi  
        time_afterpost = time()
        mnt_writer(mnt_nms, img_name[i], img_size, "%s/%s/%s.mnt"%(output_dir, set_name, img_name[i]))        
        draw_ori_on_img(image, ori, np.ones_like(seg_out), "%s/%s/%s_ori.png"%(output_dir, set_name, img_name[i]))        
        draw_minutiae(image, mnt_nms[:,:3], "%s/%s/%s_mnt.png"%(output_dir, set_name, img_name[i]))
        misc.imsave("%s/%s/%s_enh.png"%(output_dir, set_name, img_name[i]), np.squeeze(enhance_img)*ndimage.zoom(np.round(np.squeeze(seg_out)), [8,8], order=0))
        misc.imsave("%s/%s/%s_seg.png"%(output_dir, set_name, img_name[i]), ndimage.zoom(np.round(np.squeeze(seg_out)), [8,8], order=0)) 
        io.savemat("%s/%s/%s.mat"%(output_dir, set_name, img_name[i]), {'orientation':ori, 'orientation_distribution_map':ori_out_1})
        time_afterdraw = time()
        time_c.append([time_afterconv-time_start, time_afterpost-time_afterconv, time_afterdraw-time_afterpost])
        logging.info("load+conv: %.3fs, seg-postpro+nms: %.3f, draw: %.3f"%(time_c[-1][0],time_c[-1][1],time_c[-1][2]))
    time_c = np.mean(np.array(time_c),axis=0)
    logging.info("Average: load+conv: %.3fs, oir-select+seg-post+nms: %.3f, draw: %.3f"%(time_c[0],time_c[1],time_c[2]))
    return  

def main():
    if args.mode == 'train':
        train()
    elif args.mode == 'test':        
        for folder in test_set:
            test([folder,], pretrain, output_dir+"/", test_num=258, draw=False) 
    elif args.mode == 'deploy':
        for i, folder in enumerate(deploy_set):
            deploy(folder, str(i))
    else:
        pass

if __name__ =='__main__':
    main()
