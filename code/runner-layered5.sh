#!/bin/bash

#runner-layered5.sh
# runs layered-lr-modelbuilder5.py

# script, sample_len, superfactor, test_num, loud = argv


#python layered-lr-modelbuilder5.py 125 16 125 1
#python layered-lr-modelbuilder5.py 175 23 175 1




#python layered-lr-modelbuilder5.py 5 1 005 1 0.001 #<- Crashed the system!!
#python layered-lr-modelbuilder5.py 25 4 025 1 0.001
python layered-lr-modelbuilder5.py 50 8 050 1 0.002
#python layered-lr-modelbuilder5.py 75 10 075 1 0.001
python layered-lr-modelbuilder5.py 100 13 100 1 0.002
#python layered-lr-modelbuilder5.py 125 16 125 1 0.001
python layered-lr-modelbuilder5.py 150 18 150 1 0.002
#python layered-lr-modelbuilder5.py 175 23 175 1 0.001
python layered-lr-modelbuilder5.py 200 25 200 1 0.002
#python layered-lr-modelbuilder5.py 225 27 225 1 0.001
#python layered-lr-modelbuilder5.py 250 29 250 1 0.001
