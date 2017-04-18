import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#script, sample_len, superfactor, test_num = argv
from sys import argv
from datetime import date
from preprocess2 import preprocess
from helper import *
#loud = 1 means print lots of output

script, sample_len, superfactor, test_num, loud = argv

sample_len = int(sample_len)
superfactor = int(superfactor)
print("Integration TEST")
print("Sample length = %d" % (sample_len))
print("Superfactor = %d" % (superfactor))


leftvals,rightvals = preprocess(sample_len)

#samples = df.values[5:,:] #all rows, from 5th column on
leftsamples  = leftvals[5:,:] #all cols, from 5th row on
rightsamples = rightvals[5:,:] #all cols, from 5th row on

xleft = leftsamples[::3,:].T
yleft = leftsamples[1::3,:].T
zleft = leftsamples[2::3,:].T
xright = rightsamples[::3,:].T
yright = rightsamples[1::3,:].T
zright = rightsamples[2::3,:].T
outleft = leftvals[3:5,:].T
outright = rightvals[3:5,:].T
leftvals  = 0
rightvals = 0

print("xleft  shape = %s" % str(xleft.shape))
print("xright shape = %s" % str(xright.shape))

# Now we have all the samples as rows
# Next lines make same-size arrays to be overwritten with position and velocity data
xleftpos = leftsamples[::3,:].T
yleftpos = leftsamples[1::3,:].T
zleftpos = leftsamples[2::3,:].T
xrightpos = rightsamples[::3,:].T
yrightpos = rightsamples[1::3,:].T
zrightpos = rightsamples[2::3,:].T
xleftvel = leftsamples[::3,:].T
yleftvel = leftsamples[1::3,:].T
zleftvel = leftsamples[2::3,:].T
xrightvel = rightsamples[::3,:].T
yrightvel = rightsamples[1::3,:].T
zrightvel = rightsamples[2::3,:].T


# We know deltat = 0.01 because the data is at 100Hz
# I have to assume no initial velocity and the initial position is at the origin
# But I make make the initial values be the previous values in the matrices
#   --I believe that is correct but I want to plot what happens
for j in range(0,50,10):
    for t in range(0,sample_len):
        xleftvel[j][t] = 0 + xleft[j][t]*0.01
        xleftpos[j][t] = 0 + xleftvel[j][t]*0.01 + 0.5*xleft[j][t]*0.01*0.01
        yleftvel[j][t] = 0 + yleft[j][t]*0.01
        yleftpos[j][t] = 0 + yleftvel[j][t]*0.01 + 0.5*yleft[j][t]*0.01*0.01
        zleftvel[j][t] = 0 + zleft[j][t]*0.01
        zleftpos[j][t] = 0 + zleftvel[j][t]*0.01 + 0.5*zleft[j][t]*0.01*0.01

    fig = plt.figure()
    hh = "hh"
    if outleft[j][1] > 0:
        hh = "nhh"
    fig.suptitle("plot-%03d-%02dl %s" % (sample_len,j,hh))
    ax = fig.gca(projection='3d')
    ax.plot(xleftpos[j],yleftpos[j],zleftpos[j])
    plt.savefig("plot-%03d-%02dl.png" % (sample_len,j))
    
for j in range(0,50,10):
    #figure(1, figsize=(6,6))
    for t in range(0,sample_len):
        xrightvel[j][t] = 0 + xright[j][t]*0.01
        xrightpos[j][t] = 0 + xrightvel[j][t]*0.01 + 0.5*xright[j][t]*0.01*0.01
        yrightvel[j][t] = 0 + yright[j][t]*0.01
        yrightpos[j][t] = 0 + yrightvel[j][t]*0.01 + 0.5*yright[j][t]*0.01*0.01
        zrightvel[j][t] = 0 + zright[j][t]*0.01
        zrightpos[j][t] = 0 + zrightvel[j][t]*0.01 + 0.5*zright[j][t]*0.01*0.01

    fig = plt.figure()
    hh = "hh"
    if outleft[j][1] > 0:
        hh = "nhh"
    fig.suptitle("plot-%03d-%02dl %s" % (sample_len,j,hh))
    ax = fig.gca(projection='3d')
    ax.plot(xleftpos[j],yleftpos[j],zleftpos[j])
    plt.savefig("plot-%03d-%02dr.png" % (sample_len,j))
