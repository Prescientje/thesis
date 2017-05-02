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

def get_noSS_coeff(sample_len):
    if sample_len == 5:
        return 100
    if sample_len == 25:
        return 80
    if sample_len == 50:
        return 70
    if sample_len == 75:
        return 60
    if sample_len == 100:
        return 50
    if sample_len == 125:
        return 40
    if sample_len == 150:
        return 30
    if sample_len == 175:
        return 30
    if sample_len == 200:
        return 30
    if sample_len == 225:
        return 30
    if sample_len == 250:
        return 30
    else:
        print("invalid sample_len")
        return -1


def preprocess(sample_len):
    old = pd.read_csv('shadowed.csv')
    #print(old.head())
    col_names = {0: 'id', 1: 'hand', 2: 'order', 3: 'HH', 4: 'NHH', 5: 'OFF', 6: 'UNK'}
    #print(sample_len)
    superfactor = int(get_superfactor(sample_len))
    noSScoeff = int(get_noSS_coeff(sample_len))
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
    old = 0
    hhlens = []
    for j in range(len(olddata)):
        ls = olddata[(j,8)]    #8 is the proper column for the values
        line = np.array([[float(x) for x in ls[1:-1].split(', ')]])
        if olddata[(j,4)] >= 0.5 and len(line[0])>sample_len: #hh sample
            total_entries_over_sample_len += ceil((len(line[0])-sample_len+1)/superfactor)
            hhlens.append(len(line[0]))
        else:
            total_entries_over_sample_len += ceil(len(line[0])/sample_len)

    total_entries_over_sample_len = int(total_entries_over_sample_len)

    # Now we know how big to make our data frame array.
    # data is total_entries_over_sample_len
    #         by sample_len + 7 (number of header values)
    #print(total_entries_over_sample_len)
    #print(sample_len)
    data = np.zeros((total_entries_over_sample_len,7+sample_len), dtype=np.float32)
    
    #print("teosl = %d" % total_entries_over_sample_len)
    #print("sample_len = %d" % sample_len)
    offset = 0
    lhandsum = 0
    rhandsum = 0
    vals = np.zeros(len(data))
    vals2 = np.arange(len(data))
    for j in range(len(olddata)):
        ls = olddata[(j,8)]    #8 is the proper column for the values
        user = olddata[(j,1)]
        hand   = olddata[(j,2)]
        if hand > 0.5:
            rhandsum += 1
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
        #if hh >= 0.5 and len(line[0])>sample_len:
            #s = 0
            #supersample the hh samples if they are long enough
            #for k in range(0,len(line[0]) - sample_len+1,superfactor):
                #s += 1
                #vals2[offset+k//superfactor] -= offset+k//superfactor
                #data[offset+k//superfactor,7:] = line[0,k:k+sample_len]
                #data[offset+k//superfactor,0] = user
                #data[offset+k//superfactor,1] = hand
                #data[offset+k//superfactor,2] = order
                #data[offset+k//superfactor,3] = hh
                #data[offset+k//superfactor,4] = nhh
                #data[offset+k//superfactor,5] = off
                #data[offset+k//superfactor,6] = unk

            #offset += ceil((len(line[0])-sample_len+1)/superfactor)
            #print(ceil((len(line[0])-sample_len+1)/superfactor) == s)

            # Use the data if the line is divisible by 30 or it's hh
            # --> caused hand imbalance issue !!
        e = int(ceil(len(line[0])/sample_len)*sample_len)
        s = 0
        for k in range(0,e,sample_len):
            s += 1
            if k+sample_len > len(line[0]):
                data[offset+k//sample_len,7:(len(line[0])-(k+sample_len))] = line[0,k:len(line[0])]
            else:
                data[offset+k//sample_len,7:] = line[0,k:k+sample_len]
            vals2[offset+k//sample_len] -= offset+k//sample_len
            data[offset+k//sample_len,0] = user
            data[offset+k//sample_len,1] = hand
            data[offset+k//sample_len,2] = order
            data[offset+k//sample_len,3] = hh
            data[offset+k//sample_len,4] = nhh
            data[offset+k//sample_len,5] = off
            data[offset+k//sample_len,6] = unk

        offset += ceil(len(line[0])/sample_len)
        #print(ceil(len(line[0])/sample_len) == s)
    print(vals2.sum())
    print(lhandsum)
    print(rhandsum)
    print("data.shape = ", data.shape)

    #print("val diff= %d" % ((vals - val2).sum()))
    #print("val diff= %d" % (vals.sum()-val2.sum()))
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
    print("leftvals shape = %s" % str(leftvals.shape))
    print("rightvals shape = %s" % str(rightvals.shape))
    print("leftvals len = %s" % str(len(leftvals)))
    hhc = float(df['HH'].sum())
    nhhc = float(df['NHH'].sum())
    print("nhh = %f" % (nhhc/(hhc+nhhc)))
    df = 0
    '''
    for j in range(len(leftvals)):
        if j % (len(leftvals)//20) == 0:
            print(j)
        if leftvals[(j,0)] != rightvals[(j,0)]:
            print("row %d has a mismatch id" % j)
        elif leftvals[(j,2)] != rightvals[(j,2)]:
            print("row %d has a mismatch order" % j)
        if leftvals[(j,3)] != rightvals[(j,3)]:
            print("row %d has a mismatch hh" % j)
        elif leftvals[(j,4)] != rightvals[(j,4)]:
            print("row %d has a mismatch nhh" % j)
        rowid = leftvals[(j,0)]
        roworder = leftvals[(j,2)]
        rowhh = leftvals[(j,3)]
        rownhh = leftvals[(j,4)]
        for i in range(len(rightvals)):
            if rightvals[(i,0)] == rowid and rightvals[(i,2)] == roworder:
                #print(i)
                if rowhh != rightvals[(j,3)]:
                    print("row %d has a mismatch hh" % j)
                elif rownhh != rightvals[(j,4)]:
                    print("row %d has a mismatch nhh" % j)
    '''
                    
            
    #print_write("all vals in leftdf and rightdf correspond!")
    leftvals = leftvals.T
    rightvals = rightvals.T
    return leftvals,rightvals
