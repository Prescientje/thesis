import pandas as pd
import numpy as np
from math import ceil

def get_superfactor(sample_len):
    if sample_len == 5:
        return 1
    if sample_len == 25:
        return 4
    if sample_len == 50:
        return 8
    if sample_len == 64:
        return 9
    if sample_len == 75:
        return 10
    if sample_len == 100:
        return 13
    if sample_len == 125:
        return 16
    if sample_len == 144:
        return 18
    if sample_len == 150:
        return 18
    if sample_len == 175:
        return 23
    if sample_len == 200:
        return 25
    if sample_len == 225:
        return 27
    if sample_len == 250:
        return 29
    if sample_len == 256:
        return 30
    if sample_len == 576:
        return 60
    if sample_len == 900:
        return 60
    if sample_len == 1000:
        return 150
    if sample_len == 2000:
        return 240
    if sample_len == 3600:
        return 300
    else:
        print("invalid sample_len")
        return -1

def removeRandomNHH(dd1):
    print("start with check double lengths")
    leftToDelete = []
    rightToDelete = []
    leftToKeep = []
    rightToKeep = []
    for j in range(len(dd1)):
        user = dd1[j,1]
        hand = dd1[j,2]
        order = dd1[j,3]
        hh = dd1[j,4]
        nhh = dd1[j,5]
        #samples = len(np.array([[float(x) for x in dd1[j,8][1:-1].split(', ')]]))
        if (j % 50 == 0):
            print("done with %d" % j)
        for i in range(j+1,len(dd1)):
            cuser = dd1[i,1]
            chand = dd1[i,2]
            corder = dd1[i,3]
            chh = dd1[i,4]
            cnhh = dd1[i,5]
            #csamples = len(np.array([[float(x) for x in dd1[i,8][1:-1].split(', ')]]))
            #if i!=j and user==cuser and order==corder and hh==chh and nhh==cnhh and hand!=chand:
                #if samples != csamples:
                    #print("length error:",i,j,samples,csamples)
            if i!=j and user==cuser and order==corder and hh==chh and nhh==cnhh and hand!=chand and nhh>0.5:
                if hand < chand:
                    leftToDelete.append(j)
                    rightToDelete.append(i)
                    #print(j,i,hand,chand)
                else:
                    leftToDelete.append(i)
                    rightToDelete.append(j)
                    #print(i,j,chand,hand)
            if i!=j and user==cuser and order==corder and hh==chh and nhh==cnhh and hand!=chand:
                if hand < chand:
                    leftToKeep.append(j)
                    rightToKeep.append(i)
                    #print(j,i,hand,chand)
                else:
                    leftToKeep.append(i)
                    rightToKeep.append(j)
                    #print(i,j,chand,hand)

    ltd = np.array(leftToDelete)
    rtd = np.array(rightToDelete)
    for j in range(len(ltd)):
        for i in range(len(rtd)):
            if j != i and ltd[j] == rtd[i]:
                print("to delete duplicates: %d, %d" % (j,i))
    #p = np.random.permutation(len(ltd))
    #print("length of permutation = %d" % (len(p)))
    #leftdelete = ltd[p[5:]]
    #rightdelete = rtd[p[5:]]
    leftdelete  = ltd[14:]
    rightdelete = rtd[14:]
    #print("last right 5", rightdelete[-5:])
    deletedrows = np.concatenate((leftdelete,rightdelete))
    deletelc = 0
    deleterc = 0
    for j in range(len(deletedrows)):
        if dd1[deletedrows[j], 2] < 0.5:
            deletelc += 1
        if dd1[deletedrows[j], 2] > 0.5:
            deleterc += 1

    print("delete hand counts: ", deletelc, deleterc)

    #print("last total 5", deletedrows[-5:])
    res = np.delete(dd1,deletedrows,axis=0)
    print("shape of new data = %s" % str(res.shape))

    keeprows = np.concatenate((leftToKeep,rightToKeep))
    print("len of keeprows = ",len(keeprows))
    '''
    leftover = []
    for j in range(len(keeprows)):
        if j not in keeprows:
            leftover.append(j)
    res = np.delete(dd1,leftover,axis=0)
    print("shape of new data = %s" % str(res.shape))
    '''

    return res



def preprocess(sample_len):
    old = pd.read_csv('shadowed.csv')
    #print(old.head())
    col_names = {0: 'id', 1: 'hand', 2: 'order', 3: 'HH', 4: 'NHH', 5: 'OFF', 6: 'UNK'}
    #print(sample_len)
    superfactor = int(get_superfactor(sample_len))
    #noSScoeff = int(get_noSS_coeff(sample_len))
    #print("superfactor = ", superfactor)
    if superfactor < 1:
        return -1, -1
    
    # We go through each row and calculate how many rows it takes up
    # based on number of entries in the sample and how many samples
    #       we want to consider at a time
    total_entries_over_sample_len = 0
    sample_len = 3*sample_len
    #print(sample_len)
    olddata = old.values
    oldshape = olddata.shape
    olddata = removeRandomNHH(olddata)
    print("inital shape = %s" % str(oldshape))
    print("after removing shape = %s" % str(olddata.shape))
    old = 0
    hhlens = []
    for j in range(len(olddata)):
        ls = olddata[(j,8)]    #8 is the proper column for the values
        line = np.array([[float(x) for x in ls[1:-1].split(', ')]])
        #if olddata[(j,4)] >= 0.5 and len(line[0])>sample_len: #hh sample
            #total_entries_over_sample_len += ceil((len(line[0])-sample_len+1)/superfactor)
            #hhlens.append(len(line[0]))
        #else:
        total_entries_over_sample_len += ceil(len(line[0])/sample_len)

    total_entries_over_sample_len = int(total_entries_over_sample_len)

    # Now we know how big to make our data frame array.
    # data is total_entries_over_sample_len
    #         by sample_len + 7 (number of header values)
    #print(total_entries_over_sample_len)
    #print(sample_len)
    data = np.zeros((total_entries_over_sample_len,7+sample_len), dtype=np.float32)
    #checkDoubleLengths(olddata,olddata)
    
    #print("teosl = %d" % total_entries_over_sample_len)
    #print("sample_len = %d" % sample_len)
    offset = 0
    lhandsum = 0
    rhandsum = 0
    for j in range(len(olddata)):
        ls = olddata[(j,8)]    #8 is the proper column for the values
        user = olddata[(j,1)]
        hand   = olddata[(j,2)]
        if hand > 0.5:
            rhandsum += 1
            #print(j,rhandsum)
        elif hand < 0.5:
            lhandsum += 1
        else:
            print("row %d has no hand" % j)
        order  = olddata[(j,3)]
        hh  = olddata[(j,4)]
        nhh = olddata[(j,5)]
        off = olddata[(j,6)]
        unk = olddata[(j,7)]
        line = np.array([[float(x) for x in ls[1:-1].split(', ')]])

        e = int(ceil(len(line[0])/sample_len)*sample_len)
        s = 0
        for k in range(0,e,sample_len):
            s += 1
            if k+sample_len > len(line[0]):
                data[offset+k//sample_len,7:(len(line[0])-(k+sample_len))] = line[0,k:len(line[0])]
            else:
                data[offset+k//sample_len,7:] = line[0,k:k+sample_len]
            data[offset+k//sample_len,0] = user
            data[offset+k//sample_len,1] = hand
            data[offset+k//sample_len,2] = order
            data[offset+k//sample_len,3] = hh
            data[offset+k//sample_len,4] = nhh
            data[offset+k//sample_len,5] = off
            data[offset+k//sample_len,6] = unk
            #print(offset+k//sample_len) -> that part works

        offset += ceil(len(line[0])/sample_len)
        #print(ceil(len(line[0])/sample_len) == s)

    print("lhandsum=",lhandsum)
    print("rhandsum=",rhandsum)

    #So now each line is:
    #UserID / Hand / Order / HH / NHH / OFF / UNK / x0 / y0 / z0 / x1 / ...
            
    df = pd.DataFrame(data)
    data = 0 #conserve system memory
    df = df.rename(columns=col_names)
    df = df[df['OFF'] <= 0.5]
    df = df[df['UNK'] <= 0.5]
    df = df.drop('OFF', 1)
    df = df.drop('UNK', 1)
    leftdf  = df[df['hand'] <= 0.5]
    rightdf = df[df['hand'] >= 0.5]
    leftvals = leftdf.values
    rightvals = rightdf.values
    print("leftvals shape  = %s" % str(leftvals.shape))
    print("rightvals shape = %s" % str(rightvals.shape))
    print("leftvals len = %s" % str(len(leftvals)))
    hhc = float(df['HH'].sum())
    nhhc = float(df['NHH'].sum())
    print("nhh = %f" % (nhhc/(hhc+nhhc)))
    df = 0

    leftvals = leftvals.T
    rightvals = rightvals.T
    return leftvals,rightvals
