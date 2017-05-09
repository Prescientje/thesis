#!/bin/bash
# runs complex-modelbuilder.py
#script, sample_len, superfactor, test_num, loud = argv

i=4

python cmb5-noss.py 200 25 200 1 $i
python cmb5-noss.py 100 13 100 1 $i
python cmb5-noss.py 250 29 250 1 $i
python cmb5-noss.py 50 8 050 1 $i

python cmb5-noss.py 5 1 005 1 $i
python cmb5-noss.py 125 16 125 1 $i
python cmb5-noss.py 150 18 150 1 $i
python cmb5-noss.py 175 23 175 1 $i
python cmb5-noss.py 225 27 225 1 $i
python cmb5-noss.py 25 4 025 1 $i
python cmb5-noss.py 75 10 075 1 $i

#python complex-modelbuilder5.py 200 35 200 1


