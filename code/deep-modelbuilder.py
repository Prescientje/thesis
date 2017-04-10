from helper import *
from preprocess2 import preprocess

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

def getBatch(array, ix):
    if ix+batchsize < len(array[0]):
        return array[ix:ix+batchsize,:,:],ix+batchsize
    return array[ix:,:,:],0

def getBatch2(array, ix):
    if ix+batchsize < len(array):
        return array[ix:ix+batchsize,:],ix+batchsize
    return array[ix:,:],0


script, sample_len, superfactor, test_num, loud = argv
today = date.today()
file_s = "logs/" + today.strftime("%y%m%d") + "-deep-%s.txt" % test_num
left_save = sample_len + "-left/" + sample_len + "-left"
right_save = sample_len + "-right/" + sample_len + "-right"
both_save = sample_len + "-both/" + sample_len + "-both"
lmodelname = "savedModels-deep/" + left_save 
rmodelname = "savedModels-deep/" + right_save
bmodelname = "savedModels-deep/" + both_save

print()
print(file_s)
logfile = open(file_s, 'w')

sample_len = int(sample_len)
learn_rate = 0.001
stddeviation = 0.01
loud = int(loud)
superfactor = int(superfactor)
batchsize = 200
epochs = 200
print_write("Deep Modelbuilder TEST")
print_write("Uses convolutional and pooling layers.")
print_write("Sample length = %d" % (sample_len))
print_write("Learning rate = %.6f" % (learn_rate))
print_write("Superfactor = %d" % (superfactor))
print_write("Epochs = %d" % (epochs))
print_write("Batch size = %d" % (batchsize))
print_write("lmodelname = %s" % lmodelname)
print_write(" ")


leftvals,rightvals = preprocess(sample_len)
print("preprocess complete ******")
#So now each column is:
#UserID / Hand / Order / HH / NHH / OFF / UNK / x0 / y0 / z0 / x1 / ...

#samples = df.values[5:,:] #all rows, from 5th column on
'''
leftvals = leftvals.T
rightvals = rightvals.T
rightvals2 = np.zeros(rightvals.shape)
print("leftvals shape = ", leftvals.shape)
print("rightvals shape = ",rightvals.shape)
# I need to reshape these so that the entries in each row match up in time
for j in range(len(leftvals)):
    rowid = leftvals[(j,0)]
    #rowhand = leftvals[(j,1)]
    roworder = leftvals[(j,2)]
    rowhh = leftvals[(j,3)]
    rownhh = leftvals[(j,4)]
    if rightvals[(j,0)] != rowid:
        print("row %d has a mismatch id" % j)
    if rightvals[(j,2)] != rowid:
        print("row %d has a mismatch order" % j)
    if rightvals[(j,3)] != rowid:
        print("row %d has a mismatch hh" % j)
    if rightvals[(j,4)] != rowid:
        print("row %d has a mismatch nhh" % j)

rv2sum = rightvals2.sum(axis=1)
for j in range(len(rightvals2)):
    if rv2sum[j].sum() == 0:
        print("row %d has zero sum !" % j)

rightvals2 = rightvals
leftvals = leftvals.T
rightvals = rightvals.T
'''


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
outleft = leftvals[3:5,:]
print("xleft shape = %s" % str(xleft.shape))
print("outleft shape = %s" % str(outleft.shape))

xright = rightsamples[::3,:]
yright = rightsamples[1::3,:]
zright = rightsamples[2::3,:]
outright = rightvals[3:5,:]

print(sample_len)

#vvv = np.zeros([len(xleft),sample_len,3])
#for j in range(xleft.shape[0]):
    #for i in range(xleft.shape[1]):
        #vvv[j,i,:] = np.array([xleft[j,i],yleft[j,i],zleft[j,i]])
#
#print_write("vvv.shape=%s" % str(vvv.shape))

#print(xleft[0,:5])
#print(yleft[0,:5])
#print(zleft[0,:5])
#print()
#print(vvv[0,:5,:])
#vvv = np.array([xleft,yleft,zleft])

nb = int(np.ceil(len(xleft[0])/batchsize))
print_write("bs = %d" % (batchsize))
print_write("nb = %d" % (nb))

#Now each xVal col is:
#x0 / x1 / x2 / x3 / x4 / (...)
#Each output col is:
#HH / NHH

print_write(" ")
print(sample_len)
sqrtsl = int(np.sqrt(sample_len))
print_write("sqrtsl=%d" % sqrtsl)
#sample_len = int(sample_len/3) #redefine to make shape easier

# Model is softmax(x_in*weight_x + y_in*weight_y + z_in*weight_z + b)
#shape means we have the amount of values we take (sample_len)
#                 by the total number of those samples we have.

lx_place = tf.placeholder(tf.float32, [None,sample_len,3])
lx_place2 = tf.reshape(lx_place, [-1,sqrtsl,sqrtsl,3])
lout_place = tf.placeholder(tf.float32, [None,2])
print_write(str(lx_place2))
weight_lx1 = tf.Variable(tf.truncated_normal([5,5,3,32],stddev=stddeviation))
blx1        = tf.Variable(tf.truncated_normal([32],stddev=stddeviation))
lx_conv1   = tf.nn.relu(conv2d(lx_place2,weight_lx1)+blx1)
lx_pool1   = max_pool_2x2(lx_conv1)
weight_lx2 = tf.Variable(tf.truncated_normal([5,5,32,64],stddev=stddeviation))
blx2        = tf.Variable(tf.truncated_normal([64],stddev=stddeviation))
lx_conv2   = tf.nn.relu(conv2d(lx_pool1,weight_lx2)+blx2)
lx_pool2   = max_pool_2x2(lx_conv2)
weight_lxfc1  = tf.Variable(tf.truncated_normal([sqrtsl//4*sqrtsl//4*64,1024]))
blxfc1        = tf.Variable(tf.truncated_normal([1024],stddev=stddeviation))
lx_pool2_flat = tf.reshape(lx_pool2, [-1,sqrtsl//4*sqrtsl//4*64])
lx_fc1        = tf.nn.relu(tf.matmul(lx_pool2_flat, weight_lxfc1)+blxfc1)

keep_prob = tf.placeholder(tf.float32)
lx_fc1_drop = tf.nn.dropout(lx_fc1,keep_prob)

weight_lxfc2  = tf.Variable(tf.truncated_normal([1024,2]))
blxfc2        = tf.Variable(tf.truncated_normal([2],stddev=stddeviation))
lx_logits     = tf.matmul(lx_fc1_drop,weight_lxfc2)+blxfc2
lx_pred       = tf.nn.softmax(lx_logits)

lsaver = tf.train.Saver([weight_lx1, blx1, weight_lx2, blx2,
                         weight_lxfc1, blxfc2])

CE_left = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(lx_logits,lout_place))

optimizer_left = tf.train.AdamOptimizer(learning_rate=learn_rate,
                                        beta1=0.9,
                                        beta2=0.999,
                                        epsilon=1e-08).minimize(CE_left)

rx_place = tf.placeholder(tf.float32, [None,sample_len,3])
rx_place2 = tf.reshape(rx_place, [-1,sqrtsl,sqrtsl,3])
rout_place = tf.placeholder(tf.float32, [None,2])

weight_rx1 = tf.Variable(tf.truncated_normal([5,5,3,32],stddev=stddeviation))
brx1        = tf.Variable(tf.truncated_normal([32],stddev=stddeviation))
rx_conv1   = tf.nn.relu(conv2d(rx_place2,weight_rx1)+brx1)
rx_pool1   = max_pool_2x2(rx_conv1)
weight_rx2 = tf.Variable(tf.truncated_normal([5,5,32,64],stddev=stddeviation))
brx2        = tf.Variable(tf.truncated_normal([64],stddev=stddeviation))
rx_conv2   = tf.nn.relu(conv2d(rx_pool1,weight_rx2)+brx2)
rx_pool2   = max_pool_2x2(rx_conv2)
weight_rxfc1  = tf.Variable(tf.truncated_normal([sqrtsl//4*sqrtsl//4*64,1024]))
brxfc1        = tf.Variable(tf.truncated_normal([1024],stddev=stddeviation))
rx_pool2_flat = tf.reshape(rx_pool2, [-1,sqrtsl//4*sqrtsl//4*64])
rx_fc1        = tf.nn.relu(tf.matmul(rx_pool2_flat, weight_rxfc1)+brxfc1)

rx_fc1_drop = tf.nn.dropout(rx_fc1,keep_prob)

weight_rxfc2  = tf.Variable(tf.truncated_normal([1024,2]))
brxfc2        = tf.Variable(tf.truncated_normal([2],stddev=stddeviation))
rx_logits     = tf.matmul(rx_fc1_drop,weight_rxfc2)+brxfc2
rx_pred       = tf.nn.softmax(rx_logits)

rsaver = tf.train.Saver([weight_rx1, brx1, weight_rx2, brx2,
                         weight_rxfc1, brxfc2])

CE_right = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(rx_logits,rout_place))

optimizer_right = tf.train.AdamOptimizer(learning_rate=learn_rate,
                                        beta1=0.9,
                                        beta2=0.999,
                                        epsilon=1e-08).minimize(CE_right)


weight_lx1b = tf.Variable(tf.truncated_normal([5,5,3,32],stddev=stddeviation))
blx1b        = tf.Variable(tf.truncated_normal([32],stddev=stddeviation))
lx_conv1b   = tf.nn.relu(conv2d(lx_place2,weight_lx1b)+blx1b)
lx_pool1b   = max_pool_2x2(lx_conv1b)
weight_lx2b = tf.Variable(tf.truncated_normal([5,5,32,64],stddev=stddeviation))
blx2b        = tf.Variable(tf.truncated_normal([64],stddev=stddeviation))
lx_conv2b   = tf.nn.relu(conv2d(lx_pool1b,weight_lx2b)+blx2b)
lx_pool2b   = max_pool_2x2(lx_conv2b)
weight_lxfc1b  = tf.Variable(tf.truncated_normal([sqrtsl//4*sqrtsl//4*64,1024]))
blxfc1b        = tf.Variable(tf.truncated_normal([1024],stddev=stddeviation))
lx_pool2_flatb = tf.reshape(lx_pool2b, [-1,sqrtsl//4*sqrtsl//4*64])
lx_fc1b        = tf.nn.relu(tf.matmul(lx_pool2_flatb, weight_lxfc1b)+blxfc1b)
lx_fc1_dropb = tf.nn.dropout(lx_fc1b,keep_prob)
weight_lxfc2b  = tf.Variable(tf.truncated_normal([1024,2]))

weight_rx1b = tf.Variable(tf.truncated_normal([5,5,3,32],stddev=stddeviation))
brx1b        = tf.Variable(tf.truncated_normal([32],stddev=stddeviation))
rx_conv1b   = tf.nn.relu(conv2d(rx_place2,weight_rx1b)+brx1b)
rx_pool1b   = max_pool_2x2(rx_conv1b)
weight_rx2b = tf.Variable(tf.truncated_normal([5,5,32,64],stddev=stddeviation))
brx2b        = tf.Variable(tf.truncated_normal([64],stddev=stddeviation))
rx_conv2b   = tf.nn.relu(conv2d(rx_pool1b,weight_rx2b)+brx2b)
rx_pool2b   = max_pool_2x2(rx_conv2b)
weight_rxfc1b  = tf.Variable(tf.truncated_normal([sqrtsl//4*sqrtsl//4*64,1024]))
brxfc1b        = tf.Variable(tf.truncated_normal([1024],stddev=stddeviation))
rx_pool2_flatb = tf.reshape(rx_pool2b, [-1,sqrtsl//4*sqrtsl//4*64])
rx_fc1b        = tf.nn.relu(tf.matmul(rx_pool2_flatb, weight_rxfc1b)+brxfc1b)
rx_fc1_dropb = tf.nn.dropout(rx_fc1b,keep_prob)
weight_rxfc2b  = tf.Variable(tf.truncated_normal([1024,2]))

bbxfc2b        = tf.Variable(tf.truncated_normal([2],stddev=stddeviation))

bx_logits     = tf.matmul(lx_fc1_dropb,weight_lxfc2b) + tf.matmul(rx_fc1_dropb,weight_rxfc2b) + bbxfc2b
bx_pred       = tf.nn.softmax(bx_logits)



bsaver = tf.train.Saver([weight_lx1b, blx1b, weight_lx2b, blx2b,
                         weight_lxfc1b,
                         weight_rx1b, brx1b, weight_rx2b, brx2b,
                         weight_rxfc1b,
                         bbxfc2b])







init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)
print_write("init has been run")
#sess.run(optimizer_left,feed_dict={lx_place:vvv,lout_place:outleft,keep_prob:0.5})
#print_write("3d tensor optimized 1 step")

#for e in range(epochs):
    #print("epoch %d of %d deep" % (e+1,epochs))
    #ix=0
    #for i in range(nb):
        #lxbatch,_ = getBatch(vvv,ix)
        #print(lxbatch.shape)
        #outlbatch,ix = getBatch2(outleft,ix)
        #print(outlbatch.shape)
        ##sess.run([optimizer_left],feed_dict={lx_place:lxbatch,lout_place:outlbatch,keep_prob:0.5})
        #sess.run(optimizer_left,feed_dict={lx_place:vvv,lout_place:outleft,keep_prob:0.5})
        #if (i+1) % (nb/20) == 0:
            #print(i+1)

print_write("done with loop batch!")
train_scores = np.zeros([10,3])
test_scores = np.zeros([10,3])
lconfusion_sums = np.zeros([2,2])
lconfusion_sumst = np.zeros([2,2])
rconfusion_sums = np.zeros([2,2])
rconfusion_sumst = np.zeros([2,2])
bconfusion_sums = np.zeros([2,2])
bconfusion_sumst = np.zeros([2,2])
rocauc_scores = np.zeros([10,3])
minscoret = np.array([100,100,100])
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
    #print("train lxi shape = ", train_lxi.shape)
    #print("test lxi shape = ", test_lxi.shape)
    #print("train lyi shape = ", train_lyi.shape)

    aaa = np.zeros([len(train_lxi),len(train_lyi[0]),3])
    aaat = np.zeros([len(test_lxi),len(test_lyi[0]),3])
    bbb = np.zeros([len(train_ryi),len(train_ryi[0]),3])
    bbbt = np.zeros([len(test_rxi),len(test_ryi[0]),3])
    for j in range(train_lxi.shape[0]):
        for i in range(train_lxi.shape[1]):
            aaa[j,i,:] = np.array([train_lxi[j,i],train_lyi[j,i],train_lzi[j,i]])
            bbb[j,i,:] = np.array([train_rxi[j,i],train_ryi[j,i],train_rzi[j,i]])
    print_write("aaa.shape=%s" % str(aaa.shape))
    for j in range(test_lxi.shape[0]):
        for i in range(test_lxi.shape[1]):
            aaat[j,i,:] = np.array([test_lxi[j,i],test_lyi[j,i],test_lzi[j,i]])
            bbbt[j,i,:] = np.array([test_rxi[j,i],test_ryi[j,i],test_rzi[j,i]])

    nb = int(np.ceil(len(aaa)/batchsize))

    if loud > 0:
        print_write(" ")
        print_write(" ")
        print_write("Trial %d %s" % ((k+1), lmodelname))
        print_write(" ")
    
    train_ldata_feeder = {lx_place: aaa,
                          lout_place: train_louti,
                          keep_prob: 0.5}
    test_ldata_feeder = {lx_place: aaat,
                          lout_place: test_louti,
                          keep_prob: 0.5}
    train_rdata_feeder = {rx_place: bbb,
                          rout_place: train_routi,
                          keep_prob: 0.5}
    test_rdata_feeder = {rx_place: bbbt,
                          rout_place: test_routi,
                          keep_prob: 0.5}
    train_bdata_feeder = {lx_place: aaa,
                          rx_place: bbb,
                          lout_place: train_louti,
                          keep_prob: 0.5}
    test_bdata_feeder = {lx_place: aaat,
                          rx_place: bbbt,
                          lout_place: test_louti,
                          keep_prob: 0.5}

    for e in range(epochs):
        #print_write("epoch %d of %d xyz" % (e+1,epochs))
        #print_write("train len = %d" % len(aaa))
        ix = 0
        perm = np.random.permutation(len(aaa))
        for i in range(nb):
            batch_range = np.arange(ix,ix+batchsize)
            if ix+batchsize > len(aaa):
                #print("here")
                batch_range = np.arange(ix,len(aaa))
            #print(i)
            #print(batch_range[:5])
            #print(train_lxi[perm[batch_range]][0][:5])
            #print("batch shape = %s" % (str(aaa[perm[batch_range]].shape)))
            #print("batch out shape = %s" % (str(train_louti[perm[batch_range]].shape)))
            lbatch_feed = {lx_place: aaa[perm[batch_range]],
                           lout_place: train_louti[perm[batch_range]],
                           keep_prob:0.5}
            rbatch_feed = {rx_place: bbb[perm[batch_range]],
                           rout_place: train_routi[perm[batch_range]],
                           keep_prob:0.5}
            bbatch_feed = {lx_place: aaa[perm[batch_range]],
                           rx_place: bbb[perm[batch_range]],
                           lout_place: train_louti[perm[batch_range]],
                           keep_prob:0.5}

            ix += batchsize
            sess.run(optimizer_left, feed_dict=lbatch_feed)
            #sess.run(optimizer_right,feed_dict=rbatch_feed)
            #sess.run(optimizer_both, feed_dict=bbatch_feed)
            if (i+1) % (nb/20) == 0:
                print(e,i+1)
    loutputs_pred = sess.run(lx_pred, feed_dict=train_ldata_feeder)
    loutputs_test = sess.run(lx_pred, feed_dict= test_ldata_feeder)
    routputs_pred = sess.run(rx_pred, feed_dict=train_rdata_feeder)
    routputs_test = sess.run(rx_pred, feed_dict= test_rdata_feeder)
    boutputs_pred = sess.run(bx_pred, feed_dict=train_bdata_feeder)
    boutputs_test = sess.run(bx_pred, feed_dict= test_bdata_feeder)

    #calculate Xscore (train/test, Xconfusion matrix (train/test), and Xroc_auc (train/test)
    #for left, right, and both
    ls = accuracy_score(train_ldata_feeder[lout_place].argmax(axis=1),loutputs_pred.argmax(axis=1))
    lst = accuracy_score(test_ldata_feeder[lout_place].argmax(axis=1),loutputs_test.argmax(axis=1))
    lc = confusion_matrix(train_ldata_feeder[lout_place].argmax(axis=1),loutputs_pred.argmax(axis=1))
    lct = confusion_matrix(test_ldata_feeder[lout_place].argmax(axis=1),loutputs_test.argmax(axis=1))
    lconfusion_sums += lc
    lconfusion_sumst += lct
    #try:
    #    lra = roc_auc_score(train_ldata_feeder[k][lout_place].argmax(axis=1),loutputs_pred.argmax(axis=1))
    #except ValueError:
    #    lra = -1
    #try:
    #    lrat = roc_auc_score(test_ldata_feeder[k][lout_place].argmax(axis=1),loutputs_test.argmax(axis=1))
    #except ValueError:
    #    lrat = -1

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
        #print_write("ltrain roc auc score = %.6f" % (lra))
        #print_write("ltest  roc auc score = %.6f" % (lrat))
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
