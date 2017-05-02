import numpy as np
import pandas as pd
import tensorflow as tf
#script, sample_len, superfactor, test_num, loud = argv
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sys import argv
from datetime import date
from preprocessNoSuperSample import preprocess
from preprocessNoSuperSample import get_NoSS_coeff
from helper import *
#loud = 2 means print lots of output
#dir_name = './logs/'

def print_write(s):
    print(s)
    logfile.write(s)
    logfile.write("\n")

def print_confusion_vals(con):
    print_write("Miss Rate   = %0.6f" % get_missrate(con))
    print_write("Fallout     = %0.6f" % get_fallout(con))
    print_write("Precision   = %0.6f" % get_precision(con))
    print_write("Recall      = %0.6f" % get_recall(con))
    print_write("Accuracy    = %0.6f" % get_accuracy(con))
    print_write("Specificity = %0.6f" % get_specificity(con))

script, sample_len, superfactor, test_num, loud = argv
today = date.today()
file_s = "logs/" + today.strftime("%y%m%d") + "-cmb5-noss-%s.txt" % test_num
left_save = sample_len + "-left/" + sample_len + "-left"
right_save = sample_len + "-right/" + sample_len + "-right"
both_save = sample_len + "-both/" + sample_len + "-both"
lmodelname = "savedModels-complex/" + left_save 
rmodelname = "savedModels-complex/" + right_save
bmodelname = "savedModels-complex/" + both_save

print()
print(file_s)
logfile = open(file_s, 'w')

sample_len = int(sample_len)
learn_rate = 0.001
stddeviation = 0.02
ALPHA = 0.003
loud = int(loud)
batchsize = 4000
if sample_len < 51:
    batchsize = 10000
if sample_len < 24:
    batchsize = 80000
epochs = 50
superfactor = int(superfactor)
print_write("complex-modelbuilder5 TEST")
print_write("This does not save any model files.")
print_write("Sample length = %d" % (sample_len))
print_write("Learning rate = %.6f" % (learn_rate))
print_write("Stddev = %.6f" % (stddeviation))
print_write("Alpha = %f" % (ALPHA))
print_write("Superfactor = %d" % (superfactor))
print_write("Epochs = %d" % (epochs))
print_write("Batch size = %d" % (batchsize))


leftvals,rightvals = preprocess(sample_len)
print_write(" ")







'''
noSScoeff = get_NoSS_coeff(sample_len)
print(leftvals.shape)
perm = np.random.permutation(leftvals.shape[0])

find # hh samples
take that * 2.2
c = 0
for x in range (len samples)
    copy hh values, c++
for x in random permutation
    if c < len(samples2)
        add nhh sample
'''

















#samples = df.values[5:,:] #all rows, from 5th column on
leftsamples  = leftvals[5:,:] #all rows, from 5th column on
rightsamples = rightvals[5:,:] #all rows, from 5th column on

xleft = leftsamples[::3,:]
yleft = leftsamples[1::3,:]
zleft = leftsamples[2::3,:]
xright = rightsamples[::3,:]
yright = rightsamples[1::3,:]
zright = rightsamples[2::3,:]
outleft = leftvals[3:5,:]
outright = rightvals[3:5,:]
leftvals = 0
rightvals = 0

print_write("starting tensorflow section")

# Model is softmax(x_in*weight_x + y_in*weight_y + z_in*weight_z + b)
#shape means we have the amount of values we take (sample_len)
#                 by the total number of those samples we have.

lx_place = tf.placeholder(tf.float32, [None,sample_len])
ly_place = tf.placeholder(tf.float32, [None,sample_len])
lz_place = tf.placeholder(tf.float32, [None,sample_len])
lxx_place = tf.placeholder(tf.float32, [None,sample_len])
lyy_place = tf.placeholder(tf.float32, [None,sample_len])
lzz_place = tf.placeholder(tf.float32, [None,sample_len])
lxy_place = tf.placeholder(tf.float32, [None,sample_len])
lyz_place = tf.placeholder(tf.float32, [None,sample_len])
lzx_place = tf.placeholder(tf.float32, [None,sample_len])
lout_place = tf.placeholder(tf.float32, [None,2])

weight_lx = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_ly = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_lz = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_lxx = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_lyy = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_lzz = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_lxy = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_lyz = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_lzx = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
bl        = tf.Variable(tf.truncated_normal([1,2],stddev=stddeviation))
lsaver = tf.train.Saver([weight_lx, weight_ly, weight_lz,
                         weight_lxx, weight_lyy, weight_lzz,
                         weight_lxy, weight_lyz, weight_lzx, bl])

pred_left = tf.matmul(lx_place,weight_lx) + \
                                tf.matmul(ly_place,weight_ly) + \
                                tf.matmul(lz_place,weight_lz) + \
                                tf.matmul(lxx_place,weight_lxx) + \
                                tf.matmul(lyy_place,weight_lyy) + \
                                tf.matmul(lzz_place,weight_lzz) + \
                                tf.matmul(lxy_place,weight_lxy) + \
                                tf.matmul(lyz_place,weight_lyz) + \
                                tf.matmul(lzx_place,weight_lzx) + bl
prediction_left = tf.nn.softmax(pred_left)
#CE_left = tf.reduce_mean(-tf.reduce_sum(lout_place*tf.log(prediction_left), 
                                        #reduction_indices=[2]))
CE_left = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred_left,lout_place))
optimizer_left = tf.train.AdamOptimizer(learning_rate=learn_rate,
                                        beta1=0.9,
                                        beta2=0.999,
                                        epsilon=1e-08).minimize(CE_left)

rx_place = tf.placeholder(tf.float32, [None,sample_len])
ry_place = tf.placeholder(tf.float32, [None,sample_len])
rz_place = tf.placeholder(tf.float32, [None,sample_len])
rxx_place = tf.placeholder(tf.float32, [None,sample_len])
ryy_place = tf.placeholder(tf.float32, [None,sample_len])
rzz_place = tf.placeholder(tf.float32, [None,sample_len])
rxy_place = tf.placeholder(tf.float32, [None,sample_len])
ryz_place = tf.placeholder(tf.float32, [None,sample_len])
rzx_place = tf.placeholder(tf.float32, [None,sample_len])
rout_place = tf.placeholder(tf.float32, [None,2])

weight_rx = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_ry = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_rz = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_rxx = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_ryy = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_rzz = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_rxy = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_ryz = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_rzx = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
br        = tf.Variable(tf.truncated_normal([1,2],stddev=stddeviation))
rsaver = tf.train.Saver([weight_rx, weight_ry, weight_rz,
                         weight_rxx, weight_ryy, weight_rzz,
                         weight_rxy, weight_ryz, weight_rzx, br])

pred_right = tf.matmul(rx_place,weight_rx) + \
                                 tf.matmul(ry_place,weight_ry) + \
                                 tf.matmul(rz_place,weight_rz) + \
                                 tf.matmul(rxx_place,weight_rxx) + \
                                 tf.matmul(ryy_place,weight_ryy) + \
                                 tf.matmul(rzz_place,weight_rzz) + \
                                 tf.matmul(rxy_place,weight_rxy) + \
                                 tf.matmul(ryz_place,weight_ryz) + \
                                 tf.matmul(rzx_place,weight_rzx) + br
prediction_right = tf.nn.softmax(pred_right)
CE_right = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred_right,rout_place))
optimizer_right = tf.train.AdamOptimizer(learning_rate=learn_rate,
                                         beta1=0.9,
                                         beta2=0.999,
                                         epsilon=1e-08).minimize(CE_right)
'''
weight_lxb = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_lyb = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_lzb = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_lxxb = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_lyyb = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_lzzb = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_lxyb = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_lyzb = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_lzxb = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_rxb = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_ryb = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_rzb = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_rxxb = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_ryyb = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_rzzb = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_rxyb = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_ryzb = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_rzxb = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
bb        = tf.Variable(tf.truncated_normal([1,2],stddev=stddeviation))
bsaver = tf.train.Saver([weight_lxb, weight_lyb, weight_lzb,
                         weight_lxxb, weight_lyyb, weight_lzzb,
                         weight_lxyb, weight_lyzb, weight_lzxb,
                         weight_rxb, weight_ryb, weight_rzb,
                         weight_rxxb, weight_ryyb, weight_rzzb,
                         weight_rxyb, weight_ryzb, weight_rzxb, bb])
pred_both = tf.matmul(lx_place,weight_lxb) + \
            tf.matmul(ly_place,weight_lyb) + \
            tf.matmul(lz_place,weight_lzb) + \
            tf.matmul(lxx_place,weight_lxxb) + \
            tf.matmul(lyy_place,weight_lyyb) + \
            tf.matmul(lzz_place,weight_lzzb) + \
            tf.matmul(lxy_place,weight_lxyb) + \
            tf.matmul(lyz_place,weight_lyzb) + \
            tf.matmul(lzx_place,weight_lzxb) + \
            tf.matmul(rx_place,weight_rxb) + \
            tf.matmul(ry_place,weight_ryb) + \
            tf.matmul(rz_place,weight_rzb) + \
            tf.matmul(rxx_place,weight_rxxb) + \
            tf.matmul(ryy_place,weight_ryyb) + \
            tf.matmul(rzz_place,weight_rzzb) + \
            tf.matmul(rxy_place,weight_rxyb) + \
            tf.matmul(ryz_place,weight_ryzb) + \
            tf.matmul(rzx_place,weight_rzxb) + bb 
'''

wblx1, bblx1, layerbxl1 = makeLayer(lx_place,sample_len,512,"relu","wblx1","bblx1")
wbly1, bbly1, layerbyl1 = makeLayer(ly_place,sample_len,512,"relu","wbly1","bbly1")
wblz1, bblz1, layerbzl1 = makeLayer(lz_place,sample_len,512,"relu","wblz1","bblz1")
wblxx1, bblxx1, layerbxxl1 = makeLayer(lxx_place,sample_len,512,"relu","wblxx1","bblxx1")
wblyy1, bblyy1, layerbyyl1 = makeLayer(lyy_place,sample_len,512,"relu","wblyy1","bblyy1")
wblzz1, bblzz1, layerbzzl1 = makeLayer(lzz_place,sample_len,512,"relu","wblzz1","bblzz1")
wblxy1, bblxy1, layerbxyl1 = makeLayer(lxy_place,sample_len,512,"relu","wblxy1","bblxy1")
wblyz1, bblyz1, layerbyzl1 = makeLayer(lyz_place,sample_len,512,"relu","wblyz1","bblyz1")
wblzx1, bblzx1, layerbzxl1 = makeLayer(lzx_place,sample_len,512,"relu","wblzx1","bblzx1")
wbrx1, bbrx1, layerbxr1 = makeLayer(rx_place,sample_len,512,"relu","wbrx1","bbrx1")
wbry1, bbry1, layerbyr1 = makeLayer(ry_place,sample_len,512,"relu","wbry1","bbry1")
wbrz1, bbrz1, layerbzr1 = makeLayer(rz_place,sample_len,512,"relu","wbrz1","bbrz1")
wbrxx1, bbrxx1, layerbxxr1 = makeLayer(rxx_place,sample_len,512,"relu","wbrxx1","bbrxx1")
wbryy1, bbryy1, layerbyyr1 = makeLayer(ryy_place,sample_len,512,"relu","wbryy1","bbryy1")
wbrzz1, bbrzz1, layerbzzr1 = makeLayer(rzz_place,sample_len,512,"relu","wbrzz1","bbrzz1")
wbrxy1, bbrxy1, layerbxyr1 = makeLayer(rxy_place,sample_len,512,"relu","wbrxy1","bbrxy1")
wbryz1, bbryz1, layerbyzr1 = makeLayer(ryz_place,sample_len,512,"relu","wbryz1","bbryz1")
wbrzx1, bbrzx1, layerbzxr1 = makeLayer(rzx_place,sample_len,512,"relu","wbrzx1","bbrzx1")

wblx2, bblx2, layerbxl2 = makeLayer(layerbxl1,512,512,"relu","wblx2","bblx2")
wbly2, bbly2, layerbyl2 = makeLayer(layerbyl1,512,512,"relu","wbly2","bbly2")
wblz2, bblz2, layerbzl2 = makeLayer(layerbzl1,512,512,"relu","wblz2","bblz2")
wblxx2, bblxx2, layerbxxl2 = makeLayer(layerbxxl1,512,512,"relu","wblxx2","bblxx2")
wblyy2, bblyy2, layerbyyl2 = makeLayer(layerbyyl1,512,512,"relu","wblyy2","bblyy2")
wblzz2, bblzz2, layerbzzl2 = makeLayer(layerbzzl1,512,512,"relu","wblzz2","bblzz2")
wblxy2, bblxy2, layerbxyl2 = makeLayer(layerbxyl1,512,512,"relu","wblxy2","bblxy2")
wblyz2, bblyz2, layerbyzl2 = makeLayer(layerbyzl1,512,512,"relu","wblyz2","bblyz2")
wblzx2, bblzx2, layerbzxl2 = makeLayer(layerbzxl1,512,512,"relu","wblzx2","bblzx2")
wbrx2, bbrx2, layerbxr2 = makeLayer(layerbxr1,512,512,"relu","wbrx2","bbrx2")
wbry2, bbry2, layerbyr2 = makeLayer(layerbyr1,512,512,"relu","wbry2","bbry2")
wbrz2, bbrz2, layerbzr2 = makeLayer(layerbzr1,512,512,"relu","wbrz2","bbrz2")
wbrxx2, bbrxx2, layerbxxr2 = makeLayer(layerbxxr1,512,512,"relu","wbrxx2","bbrxx2")
wbryy2, bbryy2, layerbyyr2 = makeLayer(layerbyyr1,512,512,"relu","wbryy2","bbryy2")
wbrzz2, bbrzz2, layerbzzr2 = makeLayer(layerbzzr1,512,512,"relu","wbrzz2","bbrzz2")
wbrxy2, bbrxy2, layerbxyr2 = makeLayer(layerbxyr1,512,512,"relu","wbrxy2","bbrxy2")
wbryz2, bbryz2, layerbyzr2 = makeLayer(layerbyzr1,512,512,"relu","wbryz2","bbryz2")
wbrzx2, bbrzx2, layerbzxr2 = makeLayer(layerbzxr1,512,512,"relu","wbrzx2","bbrzx2")

wblx3, bblx3, layerbxl3 = makeLayer(layerbxl2,512,512,"relu","wblx3","bblx3")
wbly3, bbly3, layerbyl3 = makeLayer(layerbyl2,512,512,"relu","wbly3","bbly3")
wblz3, bblz3, layerbzl3 = makeLayer(layerbzl2,512,512,"relu","wblz3","bblz3")
wblxx3, bblxx3, layerbxxl3 = makeLayer(layerbxxl2 ,512,512,"relu","wblxx3","bblxx3")
wblyy3, bblyy3, layerbyyl3 = makeLayer(layerbyyl2 ,512,512,"relu","wblyy3","bblyy3")
wblzz3, bblzz3, layerbzzl3 = makeLayer(layerbzzl2 ,512,512,"relu","wblzz3","bblzz3")
wblxy3, bblxy3, layerbxyl3 = makeLayer(layerbxyl2 ,512,512,"relu","wblxy3","bblxy3")
wblyz3, bblyz3, layerbyzl3 = makeLayer(layerbyzl2 ,512,512,"relu","wblyz3","bblyz3")
wblzx3, bblzx3, layerbzxl3 = makeLayer(layerbzxl2 ,512,512,"relu","wblzx3","bblzx3")
wbrx3, bbrx3, layerbxr3 = makeLayer(layerbxr2,512,512,"relu","wbrx3","bbrx3")
wbry3, bbry3, layerbyr3 = makeLayer(layerbyr2,512,512,"relu","wbry3","bbry3")
wbrz3, bbrz3, layerbzr3 = makeLayer(layerbzr2,512,512,"relu","wbrz3","bbrz3")
wbrxx3, bbrxx3, layerbxxr3 = makeLayer(layerbxxr2 ,512,512,"relu","wbrxx3","bbrxx3")
wbryy3, bbryy3, layerbyyr3 = makeLayer(layerbyyr2 ,512,512,"relu","wbryy3","bbryy3")
wbrzz3, bbrzz3, layerbzzr3 = makeLayer(layerbzzr2 ,512,512,"relu","wbrzz3","bbrzz3")
wbrxy3, bbrxy3, layerbxyr3 = makeLayer(layerbxyr2 ,512,512,"relu","wbrxy3","bbrxy3")
wbryz3, bbryz3, layerbyzr3 = makeLayer(layerbyzr2 ,512,512,"relu","wbryz3","bbryz3")
wbrzx3, bbrzx3, layerbzxr3 = makeLayer(layerbzxr2 ,512,512,"relu","wbrzx3","bbrzx3")

wblx4, bblx4, layerbxl4 = makeLayer(layerbxl3,512,512,"relu","wblx4","bblx4")
wbly4, bbly4, layerbyl4 = makeLayer(layerbyl3,512,512,"relu","wbly4","bbly4")
wblz4, bblz4, layerbzl4 = makeLayer(layerbzl3,512,512,"relu","wblz4","bblz4")
wblxx4, bblxx4, layerbxxl4 = makeLayer(layerbxxl3 ,512,512,"relu","wblxx4","bblxx4")
wblyy4, bblyy4, layerbyyl4 = makeLayer(layerbyyl3 ,512,512,"relu","wblyy4","bblyy4")
wblzz4, bblzz4, layerbzzl4 = makeLayer(layerbzzl3 ,512,512,"relu","wblzz4","bblzz4")
wblxy4, bblxy4, layerbxyl4 = makeLayer(layerbxyl3 ,512,512,"relu","wblxy4","bblxy4")
wblyz4, bblyz4, layerbyzl4 = makeLayer(layerbyzl3 ,512,512,"relu","wblyz4","bblyz4")
wblzx4, bblzx4, layerbzxl4 = makeLayer(layerbzxl3 ,512,512,"relu","wblzx4","bblzx4")
wbrx4, bbrx4, layerbxr4 = makeLayer(layerbxr3,512,512,"relu","wbrx4","bbrx4")
wbry4, bbry4, layerbyr4 = makeLayer(layerbyr3,512,512,"relu","wbry4","bbry4")
wbrz4, bbrz4, layerbzr4 = makeLayer(layerbzr3,512,512,"relu","wbrz4","bbrz4")
wbrxx4, bbrxx4, layerbxxr4 = makeLayer(layerbxxr3 ,512,512,"relu","wbrxx4","bbrxx4")
wbryy4, bbryy4, layerbyyr4 = makeLayer(layerbyyr3 ,512,512,"relu","wbryy4","bbryy4")
wbrzz4, bbrzz4, layerbzzr4 = makeLayer(layerbzzr3 ,512,512,"relu","wbrzz4","bbrzz4")
wbrxy4, bbrxy4, layerbxyr4 = makeLayer(layerbxyr3 ,512,512,"relu","wbrxy4","bbrxy4")
wbryz4, bbryz4, layerbyzr4 = makeLayer(layerbyzr3 ,512,512,"relu","wbryz4","bbryz4")
wbrzx4, bbrzx4, layerbzxr4 = makeLayer(layerbzxr3 ,512,512,"relu","wbrzx4","bbrzx4")

wblx5, layerbxl5 = makeLayer(layerbxl4,512,2,None,"wblx5","bblx5")
wbly5, layerbyl5 = makeLayer(layerbyl4,512,2,None,"wbly5","bbly5")
wblz5, layerbzl5 = makeLayer(layerbzl4,512,2,None,"wblz5","bblz5")
wblxx5, layerbxxl5 = makeLayer(layerbxxl4 ,512,2,None,"wblxx5","bblxx5")
wblyy5, layerbyyl5 = makeLayer(layerbyyl4 ,512,2,None,"wblyy5","bblyy5")
wblzz5, layerbzzl5 = makeLayer(layerbzzl4 ,512,2,None,"wblzz5","bblzz5")
wblxy5, layerbxyl5 = makeLayer(layerbxyl4 ,512,2,None,"wblxy5","bblxy5")
wblyz5, layerbyzl5 = makeLayer(layerbyzl4 ,512,2,None,"wblyz5","bblyz5")
wblzx5, layerbzxl5 = makeLayer(layerbzxl4 ,512,2,None,"wblzx5","bblzx5")
wbrx5, layerbxr5 = makeLayer(layerbxr4,512,2,None,"wbrx5","bbrx5")
wbry5, layerbyr5 = makeLayer(layerbyr4,512,2,None,"wbry5","bbry5")
wbrz5, layerbzr5 = makeLayer(layerbzr4,512,2,None,"wbrz5","bbrz5")
wbrxx5, layerbxxr5 = makeLayer(layerbxxr4 ,512,2,None,"wbrxx3","bbrxx5")
wbryy5, layerbyyr5 = makeLayer(layerbyyr4 ,512,2,None,"wbryy5","bbryy5")
wbrzz5, layerbzzr5 = makeLayer(layerbzzr4 ,512,2,None,"wbrzz5","bbrzz5")
wbrxy5, layerbxyr5 = makeLayer(layerbxyr4 ,512,2,None,"wbrxy5","bbrxy5")
wbryz5, layerbyzr5 = makeLayer(layerbyzr4 ,512,2,None,"wbryz5","bbryz5")
wbrzx5, layerbzxr5 = makeLayer(layerbzxr4 ,512,2,None,"wbrzx5","bbrzx5")

bb5 = tf.Variable(tf.truncated_normal([1,2],stddev=stddeviation),name="bb5")

pred_both = layerbxl5 + layerbyl5 + layerbzl5 + \
            layerbxr5 + layerbyr5 + layerbzr5 + \
            layerbxxl5 + layerbyyl5 + layerbzzl5 + \
            layerbxxr5 + layerbyyr5 + layerbzzr5 + \
            layerbxyl5 + layerbyzl5 + layerbzxl5 + \
            layerbxyr5 + layerbyzr5 + layerbzxr5 + bb5

prediction_both = tf.nn.softmax(pred_both)
CE_both = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred_both,lout_place))

bL2 = tf.nn.l2_loss(wblx1) + tf.nn.l2_loss(wbly1) + tf.nn.l2_loss(wblz1) + \
      tf.nn.l2_loss(wbrx1) + tf.nn.l2_loss(wbry1) + tf.nn.l2_loss(wbrz1) + \
      tf.nn.l2_loss(wblxx1) + tf.nn.l2_loss(wblyy1) + tf.nn.l2_loss(wblzz1) + \
      tf.nn.l2_loss(wblxy1) + tf.nn.l2_loss(wblyz1) + tf.nn.l2_loss(wblzx1) + \
      tf.nn.l2_loss(wbrxx1) + tf.nn.l2_loss(wbryy1) + tf.nn.l2_loss(wbrzz1) + \
      tf.nn.l2_loss(wbrxy1) + tf.nn.l2_loss(wbryz1) + tf.nn.l2_loss(wbrzx1)

CE_both2 = CE_both + ALPHA*bL2


optimizer_both = tf.train.AdamOptimizer(learning_rate=learn_rate,
                                        beta1=0.9,
                                        beta2=0.999,
                                        epsilon=2e-08).minimize(CE_both2)

init = tf.initialize_all_variables()

#sess = tf.Session()
#sess.run(init)

ce_scores = np.zeros([10,3])
train_scores = np.zeros([10,3])
test_scores = np.zeros([10,3])
lconfusion_sums = np.zeros([2,2])
lconfusion_sumst = np.zeros([2,2])
rconfusion_sums = np.zeros([2,2])
rconfusion_sumst = np.zeros([2,2])
bconfusion_sums = np.zeros([2,2])
bconfusion_sumst = np.zeros([2,2])
minscoret = np.zeros(3)
for i in range(3):
    minscoret[i] = 100
maxscoret = np.zeros(3)


for k in range(10): 
    sess = tf.Session()
    sess.run(init)

    train_lxi, test_lxi = getSection(xleft,k+1)
    train_lyi, test_lyi = getSection(yleft,k+1)
    train_lzi, test_lzi = getSection(zleft,k+1)
    train_louti, test_louti = getSection(outleft,k+1)
    train_rxi, test_rxi = getSection(xright,k+1)
    train_ryi, test_ryi = getSection(yright,k+1)
    train_rzi, test_rzi = getSection(zright,k+1)
    train_routi, test_routi = getSection(outright,k+1)
    train_ldata_feeder={lx_place: train_lxi,
                               ly_place: train_lyi,
                               lz_place: train_lzi,
                               lxx_place: train_lxi*train_lxi,
                               lyy_place: train_lyi*train_lyi,
                               lzz_place: train_lzi*train_lzi,
                               lxy_place: train_lxi*train_lyi,
                               lyz_place: train_lyi*train_lzi,
                               lzx_place: train_lzi*train_lxi,
                               lout_place: train_louti}
    test_ldata_feeder={lx_place: test_lxi,
                              ly_place: test_lyi,
                              lz_place: test_lzi,
                              lxx_place: test_lxi*test_lxi,
                              lyy_place: test_lyi*test_lyi,
                              lzz_place: test_lzi*test_lzi,
                              lxy_place: test_lxi*test_lyi,
                              lyz_place: test_lyi*test_lzi,
                              lzx_place: test_lzi*test_lxi,
                              lout_place: test_louti}
    train_rdata_feeder={rx_place: train_rxi,
                               ry_place: train_ryi,
                               rz_place: train_rzi,
                               rxx_place: train_rxi*train_rxi,
                               ryy_place: train_ryi*train_ryi,
                               rzz_place: train_rzi*train_rzi,
                               rxy_place: train_rxi*train_ryi,
                               ryz_place: train_ryi*train_rzi,
                               rzx_place: train_rzi*train_rxi,
                               rout_place: train_routi}
    test_rdata_feeder={rx_place: test_rxi,
                              ry_place: test_ryi,
                              rz_place: test_rzi,
                              rxx_place: test_rxi*test_rxi,
                              ryy_place: test_ryi*test_ryi,
                              rzz_place: test_rzi*test_rzi,
                              rxy_place: test_rxi*test_ryi,
                              ryz_place: test_ryi*test_rzi,
                              rzx_place: test_rzi*test_rxi,
                              rout_place: test_routi}
    train_bdata_feeder={lx_place: train_lxi,
                               ly_place: train_lyi,
                               lz_place: train_lzi,
                               lxx_place: train_lxi*train_lxi,
                               lyy_place: train_lyi*train_lyi,
                               lzz_place: train_lzi*train_lzi,
                               lxy_place: train_lxi*train_lyi,
                               lyz_place: train_lyi*train_lzi,
                               lzx_place: train_lzi*train_lxi,
                               rx_place: train_rxi,
                               ry_place: train_ryi,
                               rz_place: train_rzi,
                               rxx_place: train_rxi*train_rxi,
                               ryy_place: train_ryi*train_ryi,
                               rzz_place: train_rzi*train_rzi,
                               rxy_place: train_rxi*train_ryi,
                               ryz_place: train_ryi*train_rzi,
                               rzx_place: train_rzi*train_rxi,
                               lout_place: train_louti}
    test_bdata_feeder={lx_place: test_lxi,
                              ly_place: test_lyi,
                              lz_place: test_lzi,
                              lxx_place: test_lxi*test_lxi,
                              lyy_place: test_lyi*test_lyi,
                              lzz_place: test_lzi*test_lzi,
                              lxy_place: test_lxi*test_lyi,
                              lyz_place: test_lyi*test_lzi,
                              lzx_place: test_lzi*test_lxi,
                              rx_place: test_rxi,
                              ry_place: test_ryi,
                              rz_place: test_rzi,
                              rxx_place: test_rxi*test_rxi,
                              ryy_place: test_ryi*test_ryi,
                              rzz_place: test_rzi*test_rzi,
                              rxy_place: test_rxi*test_ryi,
                              ryz_place: test_ryi*test_rzi,
                              rzx_place: test_rzi*test_rxi,
                              lout_place: test_louti}

    nb = int(np.ceil(len(train_lxi)/batchsize))

    if loud > 0:
        print_write(" ")
        print_write(" ")
        print_write("Trial %d %s" % ((k+1), lmodelname))
        print_write(" ")

    for e in range(epochs):
        #print("epoch %d of %d complex %d" % (e+1,epochs,sample_len))
        #print_write("train len = %d" % len(train_lxi))
        ix = 0
        perm = np.random.permutation(len(train_lxi))
        for i in range(nb):
            batch_range = np.arange(ix,ix+batchsize)
            if ix+batchsize > len(train_lxi):
                #print("here")
                batch_range = np.arange(ix,len(train_lxi))
            #print(i)
            #print(batch_range[:5])
            #print(train_lxi[perm[batch_range]][0][:5])
            lbatch_feed = {lx_place: train_lxi[perm[batch_range]],
                           ly_place: train_lyi[perm[batch_range]],
                           lz_place: train_lzi[perm[batch_range]],
                           lxx_place: train_lxi[perm[batch_range]]*train_lxi[perm[batch_range]],
                           lyy_place: train_lyi[perm[batch_range]]*train_lyi[perm[batch_range]],
                           lzz_place: train_lzi[perm[batch_range]]*train_lzi[perm[batch_range]],
                           lxy_place: train_lxi[perm[batch_range]]*train_lyi[perm[batch_range]],
                           lyz_place: train_lyi[perm[batch_range]]*train_lzi[perm[batch_range]],
                           lzx_place: train_lzi[perm[batch_range]]*train_lxi[perm[batch_range]],
                           lout_place: train_louti[perm[batch_range]]}
            rbatch_feed = {rx_place: train_rxi[perm[batch_range]],
                           ry_place: train_ryi[perm[batch_range]],
                           rz_place: train_rzi[perm[batch_range]],
                           rxx_place: train_rxi[perm[batch_range]]*train_rxi[perm[batch_range]],
                           ryy_place: train_ryi[perm[batch_range]]*train_ryi[perm[batch_range]],
                           rzz_place: train_rzi[perm[batch_range]]*train_rzi[perm[batch_range]],
                           rxy_place: train_rxi[perm[batch_range]]*train_ryi[perm[batch_range]],
                           ryz_place: train_ryi[perm[batch_range]]*train_rzi[perm[batch_range]],
                           rzx_place: train_rzi[perm[batch_range]]*train_rxi[perm[batch_range]],
                           rout_place: train_routi[perm[batch_range]]}
            bbatch_feed = {lx_place: train_lxi[perm[batch_range]],
                           ly_place: train_lyi[perm[batch_range]],
                           lz_place: train_lzi[perm[batch_range]],
                           lxx_place: train_lxi[perm[batch_range]]*train_lxi[perm[batch_range]],
                           lyy_place: train_lyi[perm[batch_range]]*train_lyi[perm[batch_range]],
                           lzz_place: train_lzi[perm[batch_range]]*train_lzi[perm[batch_range]],
                           lxy_place: train_lxi[perm[batch_range]]*train_lyi[perm[batch_range]],
                           lyz_place: train_lyi[perm[batch_range]]*train_lzi[perm[batch_range]],
                           lzx_place: train_lzi[perm[batch_range]]*train_lxi[perm[batch_range]],
                           rx_place: train_rxi[perm[batch_range]],
                           ry_place: train_ryi[perm[batch_range]],
                           rz_place: train_rzi[perm[batch_range]],
                           rxx_place: train_rxi[perm[batch_range]]*train_rxi[perm[batch_range]],
                           ryy_place: train_ryi[perm[batch_range]]*train_ryi[perm[batch_range]],
                           rzz_place: train_rzi[perm[batch_range]]*train_rzi[perm[batch_range]],
                           rxy_place: train_rxi[perm[batch_range]]*train_ryi[perm[batch_range]],
                           ryz_place: train_ryi[perm[batch_range]]*train_rzi[perm[batch_range]],
                           rzx_place: train_rzi[perm[batch_range]]*train_rxi[perm[batch_range]],
                           lout_place: train_louti[perm[batch_range]]}

            ix += batchsize
            sess.run(optimizer_left, feed_dict=lbatch_feed)
            sess.run(optimizer_right,feed_dict=rbatch_feed)
            sess.run(optimizer_both, feed_dict=bbatch_feed)
        if (e+1) % (epochs/5) == 0:
            print(e,"complex")
    loutputs_pred = sess.run(prediction_left,   feed_dict=train_ldata_feeder)
    loutputs_test = sess.run(prediction_left,   feed_dict= test_ldata_feeder)
    routputs_pred = sess.run(prediction_right,  feed_dict=train_rdata_feeder)
    routputs_test = sess.run(prediction_right,  feed_dict= test_rdata_feeder)
    boutputs_pred = sess.run(prediction_both,   feed_dict=train_bdata_feeder)
    boutputs_test = sess.run(prediction_both,   feed_dict= test_bdata_feeder)

    #calculate Xscore (train/test, Xconfusion matrix (train/test), and Xroc_auc (train/test)
    #for left, right, and both
    ls = accuracy_score(train_ldata_feeder[lout_place].argmax(axis=1),loutputs_pred.argmax(axis=1))
    lst = accuracy_score(test_ldata_feeder[lout_place].argmax(axis=1),loutputs_test.argmax(axis=1))
    lc = confusion_matrix(train_ldata_feeder[lout_place].argmax(axis=1),loutputs_pred.argmax(axis=1))
    lct = confusion_matrix(test_ldata_feeder[lout_place].argmax(axis=1),loutputs_test.argmax(axis=1))
    lconfusion_sums += lc
    lconfusion_sumst += lct
    #try:
        #lra = roc_auc_score(train_ldata_feeder[k][lout_place].argmax(axis=1),loutputs_pred.argmax(axis=1))
    #except ValueError:
        #lra = -1
    #try:
        #lrat = roc_auc_score(test_ldata_feeder[k][lout_place].argmax(axis=1),loutputs_test.argmax(axis=1))
    #except ValueError:
        #lrat = -1

    rs = accuracy_score(train_rdata_feeder[rout_place].argmax(axis=1),routputs_pred.argmax(axis=1))
    rst = accuracy_score(test_rdata_feeder[rout_place].argmax(axis=1),routputs_test.argmax(axis=1))
    rc = confusion_matrix(train_rdata_feeder[rout_place].argmax(axis=1),routputs_pred.argmax(axis=1))
    rct = confusion_matrix(test_rdata_feeder[rout_place].argmax(axis=1),routputs_test.argmax(axis=1))
    rconfusion_sums += rc
    rconfusion_sumst += rct

    bs = accuracy_score(train_bdata_feeder[lout_place].argmax(axis=1),boutputs_pred.argmax(axis=1))
    bst = accuracy_score(test_bdata_feeder[lout_place].argmax(axis=1),boutputs_test.argmax(axis=1))
    bc = confusion_matrix(train_bdata_feeder[lout_place].argmax(axis=1),boutputs_pred.argmax(axis=1))
    bct = confusion_matrix(test_bdata_feeder[lout_place].argmax(axis=1),boutputs_test.argmax(axis=1))
    bconfusion_sums += bc
    bconfusion_sumst += bct

    if loud > 0:
        print_write(" ")
        print_write("ltrain accuracy score = %f" % (ls))
        print_write("ltest  accuracy score = %f" % (lst))
        print_write("ltrain confusion matrix =\n%s" % (str(lc)))
        print_write("ltest  confusion matrix =\n%s" % (str(lct)))
        print_write("rtrain accuracy score = %f" % (rs))
        print_write("rtest  accuracy score = %f" % (rst))
        print_write("rtrain confusion matrix =\n%s" % (str(rc)))
        print_write("rtest  confusion matrix =\n%s" % (str(rct)))
        print_write("btrain accuracy score = %f" % (bs))
        print_write("btest  accuracy score = %f" % (bst))
        print_write("btrain confusion matrix =\n%s" % (str(bc)))
        print_write("btest  confusion matrix =\n%s" % (str(bct)))
        #print_write("ltrain roc auc score = %.6f" % (lra))
        #print_write("ltest  roc auc score = %.6f" % (lrat))
        print_write(" ")
    #the count of true  negatives is C_{0,0},
                 #false negatives is C_{1,0}, 
                 #true  positives is C_{1,1}, 
             #and false positives is C_{0,1}.

    train_scores[k,0] = ls
    test_scores[k,0] = lst
    train_scores[k,1] = rs
    test_scores[k,1] = rst
    train_scores[k,2] = bs
    test_scores[k,2] = bst

    if lst > maxscoret[0]:
        maxscoret[0] = lst
        #lsaver.save(sess=sess,save_path=lmodelname)
        print_write("lsaver saved %d" % (k+1))
    if lst < minscoret[0]:
        minscoret[0] = lst
    if rst > maxscoret[1]:
        maxscoret[1] = rst
        #rsaver.save(sess=sess,save_path=rmodelname)
        print_write("rsaver saved %d" % (k+1))
    if rst < minscoret[1]:
        minscoret[1] = rst
    if bst > maxscoret[2]:
        maxscoret[2] = bst
        #bsaver.save(sess=sess,save_path=bmodelname)
        print_write("bsaver saved %d" % (k+1))
    if bst < minscoret[2]:
        minscoret[2] = bst
        

    

print_write(" ")
print_write("Left Train Confusion Vals")
print_confusion_vals(lconfusion_sums)
print_write("Left Test Confusion Vals")
print_confusion_vals(lconfusion_sumst)
print_write("Right Train Confusion Vals")
print_confusion_vals(rconfusion_sums)
print_write("Right Test Confusion Vals")
print_confusion_vals(rconfusion_sumst)
print_write("Both Train Confusion Vals")
print_confusion_vals(bconfusion_sums)
print_write("Both Test Confusion Vals")
print_confusion_vals(bconfusion_sumst)
print_write("Both Test Confusion Matrix")
print_write("%s" % (str(bconfusion_sumst)))

print_write(" ")
print_write("average train score   = \n%s" % str(train_scores.mean(axis=0)))
print_write("average test score    = \n%s" % str(test_scores.mean(axis=0)))
print_write("best test score       = \n%s" % str(maxscoret))
print_write("worst test score      = \n%s" % str(minscoret))
