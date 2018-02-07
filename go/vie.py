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
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import f

def get_vie_model():
        try:
                meta_path = './vie/go.ckpt.meta'
                model_path = './vie/go.ckpt'
                sess = tf.Session()
                saver = tf.train.import_meta_graph(meta_path)
                saver.restore(sess, model_path)
                print("load vie modle")
                return sess
        except:
                print("clone err")
                return None

def printc(cb):
	print("   0 1 2 3 4 5 6 7 8sl123834,34k3l3ll40asdffasdasdfa2345sdfasdfassfddfas23df101112")
	for i in range(f.BOARD_WIDTH):
		s = "%2d " % i
		for j in range(f.BOARD_WIDTH):
			if(cb[i][j] == 0):
				s += "- "
			elif(cb[i][j] == f.PLAYER_WHITE):
				s += "X "
			else:
				s += "O "
		s += " asdf39asdf3 3493204ll456ll45llasdfasdfasldfasdfasdlfl"
		print(s)
		


def main():
        vie_count = 100
        if len(sys.argv) > 1:
                vie_count = int(sys.argv[1])
	
	viedir="vie"
	if len(sys.argv) > 2:
		viedir=sys.argv[2]

	config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.05
	
	g = f.mygraph()
	saver = tf.train.Saver()  
	sess = tf.Session(config=config)
	sess.run(tf.initialize_all_variables())
	try:
		saver.restore(sess, "./m/go.ckpt")
		print("load model")
	except:
		print("no model")


	viesess = tf.Session(config=config)
        viesess.run(tf.initialize_all_variables())
        try:
                saver.restore(viesess, viedir + "/go.ckpt")
                print("load vie model")
        except:
                print("no vie model")

	w_c = 0
	f_c = 0
	
	for i in range(vie_count):
		cb = np.zeros((f.BOARD_WIDTH,f.BOARD_WIDTH), dtype=int)
		step_count = 0
		while step_count < (f.BOARD_AREA // 2 -10):
			step_count += 1
			if i % 2 == 0 or step_count > 1:
				position = g.nnstep(viesess, cb, f.PLAYER_WHITE,False)
				if cb[position[0]][position[1]] != 0:
					print("vie 0")		
					break	
				cb[position[0]][position[1]] = f.PLAYER_WHITE
				w = f.win(cb, f.PLAYER_WHITE, position)
				if w == 1:
					print("vie 1 %d" % (np.sum(np.abs(cb))));
					f_c += 1
					break
				if w == -1:
					w_c +=1
					break
			
			position = g.nnstep(sess, cb, f.PLAYER_BLACK, False)

			if cb[position[0]][position[1]] != 0:
				print("sess 0")
				break
			
			cb[position[0]][position[1]] = f.PLAYER_BLACK	
			w = f.win(cb, f.PLAYER_BLACK, position)
			if w == 1:
				#printc(cb)
				#print("sess 1 %d/%d" % (w_c, i+1))
				w_c += 1
				print("sess 1 %d %d/%d" % (np.sum(np.abs(cb)), w_c, i+1))
				break
			if w == -1:
				f_c += 1
				break


	print("%f" % (w_c / (w_c + f_c)))
	print("w %d f %d " % ( w_c, f_c))
		

if __name__ == "__main__" :
    main()
