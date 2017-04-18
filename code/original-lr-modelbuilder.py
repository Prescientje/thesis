import numpy as np
import pandas as pd
import tensorflow as tf
#script, sample_len, superfactor, test_num, loud = argv
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
    print_write("Precision   = %0.6f" % get_precision(con))
    print_write("Recall      = %0.6f" % get_recall(con))
    print_write("Accuracy    = %0.6f" % get_accuracy(con))

script, sample_len, superfactor, test_num, loud = argv
today = date.today()
file_s = "logs/" + today.strftime("%y%m%d") + "-omb-%s.txt" % test_num
left_save = sample_len + "-0/" + sample_len + "-0"
right_save = sample_len + "-1/" + sample_len + "-1"
both_save = sample_len + "-2/" + sample_len + "-2"
lmodelname = "savedModels/" + left_save 
rmodelname = "savedModels/" + right_save
bmodelname = "savedModels/" + both_save

print()
print(file_s)
logfile = open(file_s, 'w')

sample_len = int(sample_len)
learn_rate = 0.01
stddeviation = 0.01
loud = int(loud)
batchsize = 200
epochs = 50
superfactor = int(superfactor)
print_write("ORIGINAL-LR-MODELBUILDER TEST")
print_write("Sample length = %d" % (sample_len))
print_write("Learning rate = %.3f" % (learn_rate))
print_write("Stddev = %.3f" % (stddeviation))
print_write("Superfactor = %d" % (superfactor))
print_write("Epochs = %d" % (epochs))
print_write("Batch size = %d" % (batchsize))
print_write("lmodelname = %s" % lmodelname)
print_write(" ")


leftvals,rightvals = preprocess(sample_len)


#samples = df.values[5:,:] #all rows, from 5th column on
leftsamples  = leftvals[5:,:] #all cols, from 5th row on
rightsamples = rightvals[5:,:] #all cols, from 5th row on

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


# Model is softmax(x_in*weight_x + y_in*weight_y + z_in*weight_z + b)
#shape means we have the amount of values we take (sample_len)
#                 by the total number of those samples we have.

lx_place = tf.placeholder(tf.float32, [None,sample_len])
ly_place = tf.placeholder(tf.float32, [None,sample_len])
lz_place = tf.placeholder(tf.float32, [None,sample_len])
lout_place = tf.placeholder(tf.float32, [None,2])

weight_lx = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_ly = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_lz = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
bl        = tf.Variable(tf.truncated_normal([1,2],stddev=stddeviation))

pred_left = tf.matmul(lx_place,weight_lx) + \
                                tf.matmul(ly_place,weight_ly) + \
                                tf.matmul(lz_place,weight_lz) + bl
prediction_left = tf.nn.softmax(pred_left)
CE_left = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred_left,lout_place))
optimizer_left = tf.train.AdamOptimizer(learning_rate=learn_rate,
                                        beta1=0.9,
                                        beta2=0.999,
                                        epsilon=1e-08).minimize(CE_left)

rx_place = tf.placeholder(tf.float32, [None,sample_len])
ry_place = tf.placeholder(tf.float32, [None,sample_len])
rz_place = tf.placeholder(tf.float32, [None,sample_len])
rout_place = tf.placeholder(tf.float32, [None,2])

weight_rx = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_ry = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_rz = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
br        = tf.Variable(tf.truncated_normal([1,2],stddev=stddeviation))

pred_right = tf.matmul(rx_place,weight_rx) + \
                                 tf.matmul(ry_place,weight_ry) + \
                                 tf.matmul(rz_place,weight_rz) + br
prediction_right = tf.nn.softmax(pred_right)
CE_right = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred_right,rout_place))
optimizer_right = tf.train.AdamOptimizer(learning_rate=learn_rate,
                                         beta1=0.9,
                                         beta2=0.999,
                                         epsilon=1e-08).minimize(CE_right)

weight_lxb = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_lyb = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_lzb = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_rxb = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_ryb = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
weight_rzb = tf.Variable(tf.truncated_normal([sample_len,2],stddev=stddeviation))
bb        = tf.Variable(tf.truncated_normal([1,2],stddev=stddeviation))

pred_both = tf.matmul(lx_place,weight_lxb) + \
                                tf.matmul(ly_place,weight_lyb) + \
                                tf.matmul(lz_place,weight_lzb) + \
                                tf.matmul(rx_place,weight_rxb) + \
                                tf.matmul(ry_place,weight_ryb) + \
                                tf.matmul(rz_place,weight_rzb) + bb
prediction_both = tf.nn.softmax(pred_both)
CE_both = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred_both,lout_place))
optimizer_both = tf.train.AdamOptimizer(learning_rate=learn_rate,
                                        beta1=0.9,
                                        beta2=0.999,
                                        epsilon=1e-08).minimize(CE_both)


lsaver = tf.train.Saver([weight_lx, weight_ly, weight_lz, bl])
rsaver = tf.train.Saver([weight_rx, weight_ry, weight_rz, br])
bsaver = tf.train.Saver([weight_lxb, weight_lyb, weight_lzb, 
                         weight_rxb, weight_ryb, weight_rzb, bb])
init = tf.initialize_all_variables()
#sess = tf.Session()
#sess.run(init)

#print("sess.init() has been run!")
train_scores = np.zeros([10,3])
test_scores = np.zeros([10,3])
lconfusion_sums = np.zeros([2,2])
lconfusion_sumst = np.zeros([2,2])
rconfusion_sums = np.zeros([2,2])
rconfusion_sumst = np.zeros([2,2])
bconfusion_sums = np.zeros([2,2])
bconfusion_sumst = np.zeros([2,2])
rocauc_scores = np.zeros([10,3])
minscoret = np.zeros(3)
for i in range(3):
    minscoret[i] = 100
maxscoret = np.zeros(3)



for k in range(10): 
    sess = tf.Session()
    sess.run(init)
    
    train_ldata_feeder = []
    test_ldata_feeder  = []
    train_rdata_feeder = []
    test_rdata_feeder  = []
    train_bdata_feeder = []
    test_bdata_feeder  = []
    train_lxi, test_lxi = getSection(xleft,k+1)
    train_lyi, test_lyi = getSection(yleft,k+1)
    train_lzi, test_lzi = getSection(zleft,k+1)
    train_louti, test_louti = getSection(outleft,k+1)
    train_rxi, test_rxi = getSection(xright,k+1)
    train_ryi, test_ryi = getSection(yright,k+1)
    train_rzi, test_rzi = getSection(zright,k+1)
    train_routi, test_routi = getSection(outright,k+1)
    train_ldata_feeder.append({lx_place: train_lxi,
                               ly_place: train_lyi,
                               lz_place: train_lzi,
                               lout_place: train_louti})
    test_ldata_feeder.append({lx_place: test_lxi,
                              ly_place: test_lyi,
                              lz_place: test_lzi,
                              lout_place: test_louti})
    train_rdata_feeder.append({rx_place: train_rxi,
                               ry_place: train_ryi,
                               rz_place: train_rzi,
                               rout_place: train_routi})
    test_rdata_feeder.append({rx_place: test_rxi,
                              ry_place: test_ryi,
                              rz_place: test_rzi,
                              rout_place: test_routi})
    train_bdata_feeder.append({lx_place: train_lxi,
                               ly_place: train_lyi,
                               lz_place: train_lzi,
                               rx_place: train_rxi,
                               ry_place: train_ryi,
                               rz_place: train_rzi,
                               lout_place: train_louti})
    test_bdata_feeder.append({lx_place: test_lxi,
                              ly_place: test_lyi,
                              lz_place: test_lzi,
                              rx_place: test_rxi,
                              ry_place: test_ryi,
                              rz_place: test_rzi,
                              lout_place: test_louti})
    nb = int(np.ceil(len(train_lxi)/batchsize))

    if loud > 0:
        print_write(" ")
        print_write(" ")
        print_write("Trial %d %s" % ((k+1),lmodelname))
        print_write(" ")

    for e in range(epochs):
        #print_write("epoch %d of %d original" % (e+1,epochs))
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
        if (e+1) % (epochs/5) == 0:
            print(e,"original")
        
    ''' 
    for step in range(MAXSTEPS+1):
        #(_,CEl) = sess.run([optimizer_left,CE_left],feed_dict=train_ldata_feeder[k])
        sess.run([optimizer_left],feed_dict=train_ldata_feeder[0])
        sess.run([optimizer_right],feed_dict=train_rdata_feeder[0])
        sess.run([optimizer_both],feed_dict=train_bdata_feeder[0])
    for step in range(MAXSTEPS+1):
        #(_,CEl) = sess.run([optimizer_left,CE_left],feed_dict=train_ldata_feeder[k])
        sess.run([optimizer_left],feed_dict=train_ldata_feeder[0])
        sess.run([optimizer_right],feed_dict=train_rdata_feeder[0])
        sess.run([optimizer_both],feed_dict=train_bdata_feeder[0])
    for step in range(MAXSTEPS+1):
        #(_,CEl) = sess.run([optimizer_left,CE_left],feed_dict=train_ldata_feeder[k])
        sess.run([optimizer_left],feed_dict=train_ldata_feeder[0])
        sess.run([optimizer_right],feed_dict=train_rdata_feeder[0])
        sess.run([optimizer_both],feed_dict=train_bdata_feeder[0])
        if (step+1 % (MAXSTEPS/5)) == 0:
            print(step+1)
    ''' 
    
    loutputs_pred = sess.run(prediction_left,   feed_dict=train_ldata_feeder[0])
    loutputs_test = sess.run(prediction_left,   feed_dict= test_ldata_feeder[0])
    routputs_pred = sess.run(prediction_right,  feed_dict=train_rdata_feeder[0])
    routputs_test = sess.run(prediction_right,  feed_dict= test_rdata_feeder[0])
    boutputs_pred = sess.run(prediction_both,   feed_dict=train_bdata_feeder[0])
    boutputs_test = sess.run(prediction_both,   feed_dict= test_bdata_feeder[0])

    #calculate Xscore (train/test, Xconfusion matrix (train/test), and Xroc_auc (train/test)
    #for left, right, and both
    ls = accuracy_score(train_ldata_feeder[0][lout_place].argmax(axis=1),loutputs_pred.argmax(axis=1))
    lst = accuracy_score(test_ldata_feeder[0][lout_place].argmax(axis=1),loutputs_test.argmax(axis=1))
    lc = confusion_matrix(train_ldata_feeder[0][lout_place].argmax(axis=1),loutputs_pred.argmax(axis=1))
    lct = confusion_matrix(test_ldata_feeder[0][lout_place].argmax(axis=1),loutputs_test.argmax(axis=1))
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

    rs = accuracy_score(train_rdata_feeder[0][rout_place].argmax(axis=1),routputs_pred.argmax(axis=1))
    rst = accuracy_score(test_rdata_feeder[0][rout_place].argmax(axis=1),routputs_test.argmax(axis=1))
    rc = confusion_matrix(train_rdata_feeder[0][rout_place].argmax(axis=1),routputs_pred.argmax(axis=1))
    rct = confusion_matrix(test_rdata_feeder[0][rout_place].argmax(axis=1),routputs_test.argmax(axis=1))
    rconfusion_sums += rc
    rconfusion_sumst += rct

    bs = accuracy_score(train_bdata_feeder[0][lout_place].argmax(axis=1),boutputs_pred.argmax(axis=1))
    bst = accuracy_score(test_bdata_feeder[0][lout_place].argmax(axis=1),boutputs_test.argmax(axis=1))
    bc = confusion_matrix(train_bdata_feeder[0][lout_place].argmax(axis=1),boutputs_pred.argmax(axis=1))
    bct = confusion_matrix(test_bdata_feeder[0][lout_place].argmax(axis=1),boutputs_test.argmax(axis=1))
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
