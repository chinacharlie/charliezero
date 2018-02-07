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
import ClientPlayer


def printc(cb):
	print("   0 1 2 3 4 5 6 7 8 9.0.1.2.3.4")
	for i in range(f.BOARD_WIDTH):
		s = "%2d " % i
		for j in range(f.BOARD_WIDTH):
			if(cb[i][j] == 0):
				s += "- "
			elif(cb[i][j] == f.PLAYER_WHITE):
				s += "0 "
			else:
				s += "x "
		s += ""
		print(s)
		


def main():

	p = ""

        if len(sys.argv) > 1:
                p = sys.argv[1]
	
	graph= f.mygraph()
	saver = tf.train.Saver() 
	config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.05
        sess = tf.Session(config=config) 
	sess.run(tf.initialize_all_variables())
	try:
		saver.restore(sess, "./playm/go.ckpt")
		print("load model")
	except:
		print("no model")

	wc = 0
	for i in range(100):
		samplex = []
		sampley = []
		random_step = 0
		nn_step = 0
		cb = np.zeros((f.BOARD_WIDTH,f.BOARD_WIDTH), dtype=int)
		g = ClientPlayer.Gobang()
		g.reset()
		print("new----")
		while np.sum(np.square(cb)) < (f.BOARD_AREA - 2):
			if len(p) > 0:	
				if np.sum(np.square(cb)) == 0:
					cb[7][7] = f.PLAYER_BLACK
					g.loads('2HH')
					#print(g.last())
			else:	
				if np.sum(np.square(cb)) == 0 and i % 2 == 0:
					cb[7][7] = f.PLAYER_BLACK
					'''
					g.loads('1HH2IH1II2HI1GI2GJ1JG2EL1FK2FJ1IJ2HJ1JJ2KK1HK2EJ1DJ2EI1EK2GK1HL2CG1DH')
					for m in range(f.BOARD_WIDTH):
						for n in range(f.BOARD_WIDTH):
							cb[m][n] = g.board()[m][n]
					'''
			
			if True:
			
				sx = []; sy = []; sr = []
				position = (7,7)
				r = 0
				if np.sum(np.abs(cb))  > 0:
					position = graph.nnstep(sess, cb, f.PLAYER_WHITE, True)
				
				while(cb[position[0]][position[1]] != 0):
					print("comput random step")
					pos_set = f.get_possible_pos(cb)
					position = pos_set[len(pos_set)//2]
			

				if cb[position[0]][position[1]] != 0:
					#print("c 0")
					break	
				print(position)
				cb[position[0]][position[1]] = f.PLAYER_WHITE
				g.put(position[0], position[1])
				
				if f.win(cb, f.PLAYER_WHITE, position) == 1:
					wc = wc + 1
					print("c 1 %d %d/%d" % (np.sum(np.abs(cb)), wc, i+1));
					if len(p) == 0:
						break

			if len(p) > 0:
				ip = '119.29.118.24'
        	                #print('request...')
                        	ret = ClientPlayer.send(ip, g.dumps())
				if f.win(cb, f.PLAYER_WHITE, position) == 1:
					print(ret)
					break
				g.loads(ret)
				
				#g.show()
				#print(g.last())
				cb[g.last()[0]][g.last()[1]] = f.PLAYER_BLACK
				position =(g.last()[0], g.last()[1])
				if f.win(cb, f.PLAYER_BLACK, position):
                                	#printc(cb)
	                               	print("p 1 %d" % np.sum(np.abs(cb)))
					break;
				continue
			
			printc(cb)
			position =(-1, -1)	
			while(f.isover(position)):
				try:
					input_x = raw_input("x:")
					if str(input_x) == "q": 
						os.abort()
					input_y = raw_input("y:")
					position =(int(input_x), int(input_y))
				except:
					print("continue")
			
			cb[position[0]][position[1]] = f.PLAYER_BLACK	
			if f.win(cb, f.PLAYER_BLACK, position):
				#printc(cb)
				print("p 1")
				break;

if __name__ == "__main__" :
    main()
