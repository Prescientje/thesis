import numpy as np
import pandas as pd
import tensorflow as tf
#script, sample_len, superfactor, test_num, loud, ALPHA = argv
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sys import argv
from datetime import date
from preprocess2 import preprocess
from helper import *
#loud = 1 means print lots of output
#dir_name = './logs/'

def print_write(s):
    print(s)
    logfile.write(s)
    logfile.write("\n")

def print_confusion_vals(con):
    print_write("Specificity = %0.6f" % get_specificity(con))
    print_write("Miss Rate   = %0.6f" % get_missrate(con))
    print_write("Fallout     = %0.6f" % get_fallout(con))
    print_write("Recall      = %0.6f" % get_recall(con))
    print_write("Precision   = %0.6f" % get_precision(con))
    print_write("Accuracy    = %0.6f" % get_accuracy(con))

def makeLayer(x,insize,outsize,activation,wname,bname):
    W = tf.Variable(tf.truncated_normal([insize,outsize],stddev=stddeviation),name=wname)
    if activation == "relu":
        b = tf.Variable(tf.truncated_normal([1,outsize],stddev=stddeviation),name=bname)
        return W,b,tf.nn.relu(tf.matmul(x,W) + b)
    else:
        return W,tf.matmul(x,W)

script, sample_len, superfactor, test_num, loud, ALPHA = argv
today = date.today()
file_s = "logs/" + today.strftime("%y%m%d") + "-layered5-%s.txt" % test_num
left_save = sample_len + "-left/" + sample_len + "-left"
right_save = sample_len + "-right/" + sample_len + "-right"
both_save = sample_len + "-both/" + sample_len + "-both"
lmodelname = "savedModels-layered5/" + left_save 
rmodelname = "savedModels-layered5/" + right_save
bmodelname = "savedModels-layered5/" + both_save

#hand = int(hand)
#modelname = "savedModels/" + learn_rate + "-" + hand
print()
print(file_s)
logfile = open(file_s, 'w')

sample_len = int(sample_len)
learn_rate = 0.001
stddeviation = 0.1
loud = int(loud)
batchsize = 1000
if sample_len < 51:
    batchsize = 20000

epochs = 20
ALPHA = float(ALPHA)
superfactor = int(superfactor)
print_write("Layered-LR-MODELBUILDER 5 TEST")
print_write("Sample length = %d" % (sample_len))
print_write("Learning rate = %.6f" % (learn_rate))
print_write("Stddev = %.6f" % (stddeviation))
print_write("Superfactor = %d" % (superfactor))
print_write("Epochs = %d" % (epochs))
print_write("Batch size = %d" % (batchsize))
print_write("Alpha = %f" % (ALPHA))
print_write("lmodelname = %s" % lmodelname)
print_write(" ")


leftvals,rightvals = preprocess(sample_len)

#So now each column is:
#UserID / Hand / Order / HH / NHH / OFF / UNK / x0 / y0 / z0 / x1 / ...

#samples = df.values[5:,:] #all rows, from 5th column on
leftsamples  = leftvals[5:,:] #all rows, from 5th column on
rightsamples = rightvals[5:,:] #all rows, from 5th column on

#print_write("samples shape = %s" % (samples.shape,))
#xVals = samples[::3,:]
#yVals = samples[1::3,:]
#zVals = samples[2::3,:]
#outputs = df.values[3:5,:]

xleft = leftsamples[::3,:]
yleft = leftsamples[1::3,:]
zleft = leftsamples[2::3,:]
xright = rightsamples[::3,:]
yright = rightsamples[1::3,:]
zright = rightsamples[2::3,:]
outleft = leftvals[3:5,:]
outright = rightvals[3:5,:]

leftvals=0
rightvals=0

# Model is softmax(x_in*weight_x + y_in*weight_y + z_in*weight_z + b)
#shape means we have the amount of values we take (sample_len)
#                 by the total number of those samples we have.

lx_place = tf.placeholder(tf.float32, [None,sample_len])
ly_place = tf.placeholder(tf.float32, [None,sample_len])
lz_place = tf.placeholder(tf.float32, [None,sample_len])
lout_place = tf.placeholder(tf.float32, [None,2])
rx_place = tf.placeholder(tf.float32, [None,sample_len])
ry_place = tf.placeholder(tf.float32, [None,sample_len])
rz_place = tf.placeholder(tf.float32, [None,sample_len])
rout_place = tf.placeholder(tf.float32, [None,2])

weight_lx1, blx1, layerl1x = makeLayer(lx_place,sample_len,512,"relu","weight_lx1","blx1")
weight_ly1, bly1, layerl1y = makeLayer(ly_place,sample_len,512,"relu","weight_ly1","bly1")
weight_lz1, blz1, layerl1z = makeLayer(lz_place,sample_len,512,"relu","weight_lz1","blz1")
weight_lx2, layerl2x = makeLayer(layerl1x,512,2,None,"weight_lx2","blx2")
weight_ly2, layerl2y = makeLayer(layerl1y,512,2,None,"weight_lx2","blx2")
weight_lz2, layerl2z = makeLayer(layerl1z,512,2,None,"weight_lx2","blx2")
bl2        = tf.Variable(tf.truncated_normal([1,2],stddev=stddeviation))
layl2 = layerl2x + \
        layerl2y + \
        layerl2z + bl2

'''
weight_lx1 = tf.Variable(tf.truncated_normal([sample_len,512],stddev=stddeviation))
weight_ly1 = tf.Variable(tf.truncated_normal([sample_len,512],stddev=stddeviation))
weight_lz1 = tf.Variable(tf.truncated_normal([sample_len,512],stddev=stddeviation))
blx1        = tf.Variable(tf.truncated_normal([1,512],stddev=stddeviation))
bly1        = tf.Variable(tf.truncated_normal([1,512],stddev=stddeviation))
blz1        = tf.Variable(tf.truncated_normal([1,512],stddev=stddeviation))
weight_lx2 = tf.Variable(tf.truncated_normal([512,2],stddev=stddeviation))
weight_ly2 = tf.Variable(tf.truncated_normal([512,2],stddev=stddeviation))
weight_lz2 = tf.Variable(tf.truncated_normal([512,2],stddev=stddeviation))
bl2        = tf.Variable(tf.truncated_normal([1,2],stddev=stddeviation))

layerl1x = tf.nn.relu(tf.matmul(lx_place,weight_lx1) + blx1)
layerl1y = tf.nn.relu(tf.matmul(ly_place,weight_ly1) + bly1)
layerl1z = tf.nn.relu(tf.matmul(lz_place,weight_lz1) + blz1)
layl2 = tf.matmul(layerl1x,weight_lx2) + \
                        tf.matmul(layerl1y,weight_ly2) + \
                        tf.matmul(layerl1z,weight_lz2) + bl2
'''
layerl2 = tf.nn.softmax(layl2)
CE_left = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(layl2,lout_place))
optimizer_left = tf.train.AdamOptimizer(learning_rate=learn_rate,
                                        beta1=0.9,
                                        beta2=0.999,
                                        epsilon=1e-08).minimize(CE_left)

weight_rx1 = tf.Variable(tf.truncated_normal([sample_len,512],stddev=stddeviation))
weight_ry1 = tf.Variable(tf.truncated_normal([sample_len,512],stddev=stddeviation))
weight_rz1 = tf.Variable(tf.truncated_normal([sample_len,512],stddev=stddeviation))
brx1        = tf.Variable(tf.truncated_normal([1,512],stddev=stddeviation))
bry1        = tf.Variable(tf.truncated_normal([1,512],stddev=stddeviation))
brz1        = tf.Variable(tf.truncated_normal([1,512],stddev=stddeviation))
weight_rx2 = tf.Variable(tf.truncated_normal([512,2],stddev=stddeviation))
weight_ry2 = tf.Variable(tf.truncated_normal([512,2],stddev=stddeviation))
weight_rz2 = tf.Variable(tf.truncated_normal([512,2],stddev=stddeviation))
br2        = tf.Variable(tf.truncated_normal([1,2],stddev=stddeviation))

layerr1x = tf.nn.relu(tf.matmul(rx_place,weight_rx1) + brx1)
layerr1y = tf.nn.relu(tf.matmul(ry_place,weight_ry1) + bry1)
layerr1z = tf.nn.relu(tf.matmul(rz_place,weight_rz1) + brz1)
layr2 = tf.matmul(layerr1x,weight_rx2) + \
        tf.matmul(layerr1y,weight_ry2) + \
        tf.matmul(layerr1z,weight_rz2) + br2
layerr2 = tf.nn.softmax(layr2)
CE_right = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(layr2,rout_place))
optimizer_right = tf.train.AdamOptimizer(learning_rate=learn_rate,
                                        beta1=0.9,
                                        beta2=0.999,
                                        epsilon=1e-08).minimize(CE_right)



wblx1, bblx1, layerbxl1 = makeLayer(lx_place,sample_len,512,"relu","wblx1","bblx1")
wbly1, bbly1, layerbyl1 = makeLayer(ly_place,sample_len,512,"relu","wbly1","bbly1")
wblz1, bblz1, layerbzl1 = makeLayer(lz_place,sample_len,512,"relu","wblz1","bblz1")
wbrx1, bbrx1, layerbxr1 = makeLayer(rx_place,sample_len,512,"relu","wbrx1","bbrx1")
wbry1, bbry1, layerbyr1 = makeLayer(ry_place,sample_len,512,"relu","wbry1","bbry1")
wbrz1, bbrz1, layerbzr1 = makeLayer(rz_place,sample_len,512,"relu","wbrz1","bbrz1")

wblx2, bblx2, layerbxl2 = makeLayer(layerbxl1,512,512,"relu","wblx2","bblx2")
wbly2, bbly2, layerbyl2 = makeLayer(layerbyl1,512,512,"relu","wbly2","bbly2")
wblz2, bblz2, layerbzl2 = makeLayer(layerbzl1,512,512,"relu","wblz2","bblz2")
wbrx2, bbrx2, layerbxr2 = makeLayer(layerbxr1,512,512,"relu","wbrx2","bbrx2")
wbry2, bbry2, layerbyr2 = makeLayer(layerbyr1,512,512,"relu","wbry2","bbry2")
wbrz2, bbrz2, layerbzr2 = makeLayer(layerbzr1,512,512,"relu","wbrz2","bbrz2")

wblx3, bblx3, layerbxl3 = makeLayer(layerbxl2,512,512,"relu","wblx3","bblx3")
wbly3, bbly3, layerbyl3 = makeLayer(layerbyl2,512,512,"relu","wbly3","bbly3")
wblz3, bblz3, layerbzl3 = makeLayer(layerbzl2,512,512,"relu","wblz3","bblz3")
wbrx3, bbrx3, layerbxr3 = makeLayer(layerbxr2,512,512,"relu","wbrx3","bbrx3")
wbry3, bbry3, layerbyr3 = makeLayer(layerbyr2,512,512,"relu","wbry3","bbry3")
wbrz3, bbrz3, layerbzr3 = makeLayer(layerbzr2,512,512,"relu","wbrz3","bbrz3")

wblx4, bblx4, layerbxl4 = makeLayer(layerbxl3,512,512,"relu","wblx4","bblx4")
wbly4, bbly4, layerbyl4 = makeLayer(layerbyl3,512,512,"relu","wbly4","bbly4")
wblz4, bblz4, layerbzl4 = makeLayer(layerbzl3,512,512,"relu","wblz4","bblz4")
wbrx4, bbrx4, layerbxr4 = makeLayer(layerbxr3,512,512,"relu","wbrx4","bbrx4")
wbry4, bbry4, layerbyr4 = makeLayer(layerbyr3,512,512,"relu","wbry4","bbry4")
wbrz4, bbrz4, layerbzr4 = makeLayer(layerbzr3,512,512,"relu","wbrz4","bbrz4")

'''
wblx5, bblx5, layerbxl5 = makeLayer(layerbxl4,512,512,"relu","wblx5","bblx5")
wbly5, bbly5, layerbyl5 = makeLayer(layerbyl4,512,512,"relu","wbly5","bbly5")
wblz5, bblz5, layerbzl5 = makeLayer(layerbzl4,512,512,"relu","wblz5","bblz5")
wbrx5, bbrx5, layerbxr5 = makeLayer(layerbxr4,512,512,"relu","wbrx5","bbrx5")
wbry5, bbry5, layerbyr5 = makeLayer(layerbyr4,512,512,"relu","wbry5","bbry5")
wbrz5, bbrz5, layerbzr5 = makeLayer(layerbzr4,512,512,"relu","wbrz5","bbrz5")
'''

wblx6, layerbxl6 = makeLayer(layerbxl4,512,2,None,"wblx6","bblx6")
wbly6, layerbyl6 = makeLayer(layerbyl4,512,2,None,"wbly6","bbly6")
wblz6, layerbzl6 = makeLayer(layerbzl4,512,2,None,"wblz6","bblz6")
wbrx6, layerbxr6 = makeLayer(layerbxr4,512,2,None,"wbrx6","bbrx6")
wbry6, layerbyr6 = makeLayer(layerbyr4,512,2,None,"wbry6","bbry6")
wbrz6, layerbzr6 = makeLayer(layerbzr4,512,2,None,"wbrz6","bbrz6")
bb6 = tf.Variable(tf.truncated_normal([1,2],stddev=stddeviation),name="bb6")

layb2 = layerbxl6 + layerbyl6 + layerbzl6 + \
        layerbxr6 + layerbyr6 + layerbzr6 + bb6

layerb2 = tf.nn.softmax(layb2)
CE_both = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(layb2,lout_place))

bL2 = tf.nn.l2_loss(wblx1)+ tf.nn.l2_loss(wbly1)+ tf.nn.l2_loss(wblz1)+ tf.nn.l2_loss(wbrx1)+ tf.nn.l2_loss(wbry1)+ tf.nn.l2_loss(wbrz1)+\
      tf.nn.l2_loss(wblx2)+ tf.nn.l2_loss(wbly2)+ tf.nn.l2_loss(wblz2)+ tf.nn.l2_loss(wbrx2)+ tf.nn.l2_loss(wbry2)+ tf.nn.l2_loss(wbrz2)+\
      tf.nn.l2_loss(wblx3)+ tf.nn.l2_loss(wbly3)+ tf.nn.l2_loss(wblz3)+ tf.nn.l2_loss(wbrx3)+ tf.nn.l2_loss(wbry3)+ tf.nn.l2_loss(wbrz3)+\
      tf.nn.l2_loss(wblx4)+ tf.nn.l2_loss(wbly4)+ tf.nn.l2_loss(wblz4)+ tf.nn.l2_loss(wbrx4)+ tf.nn.l2_loss(wbry4)+ tf.nn.l2_loss(wbrz4)+\
      tf.nn.l2_loss(wblx6)+ tf.nn.l2_loss(wbly6)+ tf.nn.l2_loss(wblz6)+ tf.nn.l2_loss(wbrx6)+ tf.nn.l2_loss(wbry6)+ tf.nn.l2_loss(wbrz6)
CE_both2 = CE_both + ALPHA*bL2
optimizer_both = tf.train.AdamOptimizer(learning_rate=learn_rate,
                                        beta1=0.9,
                                        beta2=0.999,
                                        epsilon=1e-08).minimize(CE_both2)


lsaver = tf.train.Saver([weight_lx1, weight_ly1, weight_lz1,
                         blx1, bly1, blz1,
                         weight_lx2, weight_ly2, weight_lz2,
                         bl2])
rsaver = tf.train.Saver([weight_rx1, weight_ry1, weight_rz1,
                         brx1, bry1, brz1,
                         weight_rx2, weight_ry2, weight_rz2,
                         br2])

bsaver = tf.train.Saver([wblx1, wbly1, wblz1, wbrx1, wbry1, wbrz1,
                         bblx1, bbly1, bblz1, bbrx1, bbry1, bbrz1,
                         wblx2, wbly2, wblz2, wbrx2, wbry2, wbrz2,
                         bblx2, bbly2, bblz2, bbrx2, bbry2, bbrz2,
                         wblx3, wbly3, wblz3, wbrx3, wbry3, wbrz3,
                         bblx3, bbly3, bblz3, bbrx3, bbry3, bbrz3,
                         wblx4, wbly4, wblz4, wbrx4, wbry4, wbrz4,
                         bblx4, bbly4, bblz4, bbrx4, bbry4, bbrz4,
                         wblx6, wbly6, wblz6, wbrx6, wbry6, wbrz6])



init = tf.initialize_all_variables()
#sess = tf.Session()
#sess.run(init)

train_scores = np.zeros([10,3])
test_scores = np.zeros([10,3])
minscoret = np.zeros(3)
lconfusion_sums = np.zeros([2,2])
lconfusion_sumst = np.zeros([2,2])
rconfusion_sums = np.zeros([2,2])
rconfusion_sumst = np.zeros([2,2])
bconfusion_sums = np.zeros([2,2])
bconfusion_sumst = np.zeros([2,2])
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
    train_ldata_feeder = {lx_place: train_lxi,
                          ly_place: train_lyi,
                          lz_place: train_lzi,
                          lout_place: train_louti}
    test_ldata_feeder = {lx_place: test_lxi,
                         ly_place: test_lyi,
                         lz_place: test_lzi,
                         lout_place: test_louti}
    train_rdata_feeder = {rx_place: train_rxi,
                          ry_place: train_ryi,
                          rz_place: train_rzi,
                          rout_place: train_routi}
    test_rdata_feeder = {rx_place: test_rxi,
                         ry_place: test_ryi,
                         rz_place: test_rzi,
                         rout_place: test_routi}
    train_bdata_feeder = {lx_place: train_lxi,
                          ly_place: train_lyi,
                          lz_place: train_lzi,
                          rx_place: train_rxi,
                          ry_place: train_ryi,
                          rz_place: train_rzi,
                          lout_place: train_louti}
    test_bdata_feeder = {lx_place: test_lxi,
                         ly_place: test_lyi,
                         lz_place: test_lzi,
                         rx_place: test_rxi,
                         ry_place: test_ryi,
                         rz_place: test_rzi,
                         lout_place: test_louti}
    nb = int(np.ceil(len(train_lxi)/batchsize))
    if loud > 0:
        print_write(" ")
        print_write(" ")
        print_write("Trial %d %s" % ((k+1),lmodelname))
        print_write(" ")

    for e in range(epochs):
        #print_write("epoch %d of %d layered" % (e+1,epochs))
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
                           lout_place: train_louti[perm[batch_range]]}
            rbatch_feed = {rx_place: train_rxi[perm[batch_range]],
                           ry_place: train_ryi[perm[batch_range]],
                           rz_place: train_rzi[perm[batch_range]],
                           rout_place: train_routi[perm[batch_range]]}
            bbatch_feed = {lx_place: train_lxi[perm[batch_range]],
                           ly_place: train_lyi[perm[batch_range]],
                           lz_place: train_lzi[perm[batch_range]],
                           rx_place: train_rxi[perm[batch_range]],
                           ry_place: train_ryi[perm[batch_range]],
                           rz_place: train_rzi[perm[batch_range]],
                           lout_place: train_louti[perm[batch_range]]}

            ix += batchsize
            sess.run(optimizer_left, feed_dict=lbatch_feed)
            sess.run(optimizer_right,feed_dict=rbatch_feed)
            sess.run(optimizer_both, feed_dict=bbatch_feed)
        if (e+1) % (epochs/10) == 0:
            #print(e+1,"layered5")
            (bot, ceb2_test) = sess.run([layerb2,CE_both2], feed_dict= test_bdata_feeder)
            bat = accuracy_score(test_bdata_feeder[lout_place].argmax(axis=1),bot.argmax(axis=1))
            print_write("epoch %d: ce_both = %f, acc test both = %f" % (e+1,ceb2_test,bat))

    loutputs_pred = sess.run(layerl2, feed_dict=train_ldata_feeder)
    loutputs_test = sess.run(layerl2, feed_dict= test_ldata_feeder)
    routputs_pred = sess.run(layerr2, feed_dict=train_rdata_feeder)
    routputs_test = sess.run(layerr2, feed_dict= test_rdata_feeder)
    boutputs_pred = sess.run(layerb2, feed_dict=train_bdata_feeder)
    boutputs_test = sess.run(layerb2, feed_dict= test_bdata_feeder)

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
        print_write(" ")
        print_write("ltrain accuracy score = %f" % (ls))
        print_write("ltest  accuracy score = %f" % (lst))
        print_write("ltrain confusion matrix =\n%s" % (str(lc)))
        print_write("ltest  confusion matrix =\n%s" % (str(lct)))
        print_write(" ")
        print_write("rtrain accuracy score = %f" % (rs))
        print_write("rtest  accuracy score = %f" % (rst))
        print_write("rtrain confusion matrix =\n%s" % (str(rc)))
        print_write("rtest  confusion matrix =\n%s" % (str(rct)))
        print_write(" ")
        print_write("btrain accuracy score = %f" % (bs))
        print_write("btest  accuracy score = %f" % (bst))
        print_write("btrain confusion matrix =\n%s" % (str(bc)))
        print_write("btest  confusion matrix =\n%s" % (str(bct)))
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
        lsaver.save(sess=sess,save_path=lmodelname)
        print_write("lsaver saved %d" % (k+1))
    elif lst < minscoret[0]:
        minscoret[0] = lst
    if rst > maxscoret[1]:
        maxscoret[1] = rst
        rsaver.save(sess=sess,save_path=rmodelname)
        print_write("rsaver saved %d" % (k+1))
    elif rst < minscoret[1]:
        minscoret[1] = rst
    if bst > maxscoret[2]:
        maxscoret[2] = bst
        bsaver.save(sess=sess,save_path=bmodelname)
        print_write("bsaver saved %d" % (k+1))
    elif bst < minscoret[2]:
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
