import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#script, sample_len, superfactor, test_num = argv
from sys import argv
from datetime import date
from preprocess2 import preprocess
from helper import *
#loud = 1 means print lots of output
#dir_name = './logs/'

def print_write(s):
    print(s)
    #logfile.write(s)
    #logfile.write("\n")

script, sample_len, superfactor, test_num, loud = argv
today = date.today()
file_s = "logs/" + today.strftime("%y%m%d") + "-integrate-%s.txt" % test_num

print()
print(file_s)
#logfile = open(file_s, 'w')

sample_len = int(sample_len)
superfactor = int(superfactor)
print_write("Integration TEST")
print_write("Sample length = %d" % (sample_len))
print_write("Superfactor = %d" % (superfactor))


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

leftsamples  = 0
rightsamples = 0

# We know deltat = 0.01 because the data is at 100Hz
# I have to assume no initial velocity and the initial position is at the origin
# But I make make the initial values be the previous values in the matrices-- I believe that is correct but I want to plot what happens
for j in range(1):
    figure(1, figsize=(6,6))
    for t in range(0,sample_len):
        xleftvel[j][t] = 0 + xleft[j][t]*0.01
        xleftpos[j][t] = 0 + xleftvel[j][t]*0.01 + 0.5*xleft[j][t]*0.01*0.01

    plt.plot(xleftpos[j])
    plt.savefig("test-%02d.png" % j)
