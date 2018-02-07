# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
# ==============================================================================
#!python
"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import gzip
import os
import sys
import tempfile
import numpy as np
import pickle
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import f
import time


def savesample(xs, ys, rs, index_pos, deep):
	id = str(os.getpid()) + "_" + str(index_pos) + "_" + str(deep)
	with open('samples/xs.' + id, 'wb') as f:
		pickle.dump(xs, f)
	with open('samples/ys.' + id, 'wb') as f:
		pickle.dump(ys, f)
	with open('samples/rs.' + id, 'wb') as f:
		pickle.dump(rs, f)

def loadpart(index_id):
        xs = [];ys = [];rs = []
        try:
                with open('samples/xs.' + index_id, 'rb') as f:
                        xs = pickle.load(f)
                with open('samples/ys.' + index_id, 'rb') as f:
                        ys = pickle.load(f)
                with open('samples/rs.' + index_id, 'rb') as f:
                        rs = pickle.load(f)
        except:
                print("load error")

        return (xs, ys, rs)

def loadsamples():
	xs=[]; ys=[]; rs=[];
        files= os.listdir("samples")
        for f in files:
		print(f)
                (h, ext) = f.split('.')
		print(h)
                if h == "xs":
                        (x, y, r) = loadpart(ext)	
			xs.extend(x)
			ys.extend(y)
			rs.extend(r)
	return (xs, ys, rs)		
	
def main():
        if len(sys.argv) > 1:
                deep = int(sys.argv[1])
                if deep >= 5:
                        deep_count = 5	
	g = f.mygraph()
	saver = tf.train.Saver() 
	config = tf.ConfigProto() 
	config.gpu_options.allow_growth = True 
        #config.gpu_options.per_process_gpu_memory_fraction = 0.05
        sess = tf.Session(config=config) 
	sess.run(tf.initialize_all_variables())
	try:
		saver.restore(sess, "./m/go.ckpt")
		print("load model")
	except:
		print("no model")
	(xs, ys, rs) = f.play(sess, g, 100)
	savesample(xs, ys, rs, 0, 0)	
		
	print("ok")	
	#time.sleep(500)
if __name__ == "__main__" :
    main()
