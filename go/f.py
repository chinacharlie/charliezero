#Copyright 2015 The TensorFlow Authors. All Rights Reserved.
# ==============================================================================

"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import gzip
import os
import sys
import time
import pickle
import tempfile
import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

BOARD_WIDTH = 11
BOARD_AREA = BOARD_WIDTH * BOARD_WIDTH
PLAYER_SPACE = 0
PLAYER_BLACK = -1
PLAYER_WHITE = 1
layer_count = 6
channel_count = 64


def printboard(cb):
	print(cb)


def isover(position):
	if position[0] < 0 or position[0] >= BOARD_WIDTH:
		return True
	if position[1] < 0 or position[1] >= BOARD_WIDTH:
		return True
	return False

def get_linknum(chessboard, player, position, delta):
        linknum = 0
	x, y = position
        while (not isover((x, y)) and (chessboard[x][y]==player)):
                linknum += 1
                x += int(delta[0]);
                y += int(delta[1]);

	x = position[0] - delta[0]
	y = position[1] - delta[1]
        while (not isover((x,y)) and (chessboard[x][y]==player)):
                linknum += 1
                x -= delta[0];
                y -= delta[1];
	
	return linknum	

def win(chessboard, player, position):
	delta = ((1, 0), (1,1), (0, 1), (1, -1))
	for i in delta:
		l = get_linknum(chessboard, player, position, i)
		if l >= 5 :
			return 1
	return 0

class mygraph(object):
        def _weight_variable(self, shape):
                initial = tf.random_normal(shape, stddev=0.01, dtype=tf.float32)
                return tf.Variable(initial)
        def _bias_variable(self, shape):
                initial = tf.constant(0.01, shape=shape, dtype=tf.float32)
                return tf.Variable(initial)
        def _conv2d(self, x, W):
                return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
        def x(self):
                return self._x
        def y(self):
                return self._y
        def feed_y(self):
                return self._feed_y
        def feed_r(self):
                return self._feed_r
        def loss(self):
                return self._loss
        def train(self):
                return self._train_step
	def acc(self):
		return self._acc
        def graph(self):
                return self._graph
	def testloss(self, sess, tx, ty, tr):
        	if len(tx) > 0:
                	l = sess.run(self.loss(), feed_dict = {self.x() : tx, self.feed_y() : ty, self.feed_r() : tr})
	                a = sess.run(self.acc(), feed_dict = {self.x() : tx, self.feed_y() : ty, self.feed_r() : tr})
                	return (l, a)
		return (0, 0)

        def __init__(self):
                self._graph = tf.Graph()
                x = tf.placeholder(tf.float32, [None, BOARD_WIDTH, BOARD_WIDTH], name="x")
                x_image = tf.reshape(x, [-1, BOARD_WIDTH, BOARD_WIDTH, 1])
                x_space = tf.reshape(tf.square(1 - tf.square(x)), [-1, BOARD_WIDTH, BOARD_WIDTH, 1])
                x_relu = tf.reshape(tf.nn.relu(x), [-1, BOARD_WIDTH, BOARD_WIDTH,1])
                x_negative_relu = tf.reshape(tf.nn.relu(-x), [-1, BOARD_WIDTH, BOARD_WIDTH,1])
                x_cat = tf.concat([x_image, x_space, x_relu, x_negative_relu], 3)
                x_join = tf.reshape(x_cat, [-1, BOARD_WIDTH, BOARD_WIDTH, 4])

                w_conv = self._weight_variable([3, 3, 4, channel_count])
                b_conv =  self._bias_variable([channel_count])
                h_conv = tf.nn.relu(self._conv2d(x_join, w_conv) + b_conv)

                h_input = h_conv#x_join
                for i in range(layer_count):
                        w_conv = self._weight_variable([3, 3, channel_count, channel_count])
                        b_conv = self._bias_variable([channel_count])
                        h_conv = tf.nn.relu(h_input + self._conv2d(h_input, w_conv) + b_conv)
                        h_input = h_conv

		w_conv = self._weight_variable([1, 1, channel_count, 1])
		b_conv = self._bias_variable([1])
		h_conv = tf.nn.relu(self._conv2d(h_input, w_conv) + b_conv)
                h_input = h_conv

                h_flat = tf.reshape(h_input, [-1, BOARD_AREA])
                h_fc = h_flat

                y = tf.nn.softmax(h_fc - tf.reshape(x_relu + x_negative_relu, [-1, BOARD_AREA]) * (h_fc + 1000))
                feed_y = tf.placeholder(tf.float32, [None, BOARD_AREA], name="y_")
                feed_r = tf.placeholder(tf.float32, [None, 1], name="r_")

                cross_entropy = - tf.reduce_sum(feed_y * feed_r * tf.log(y + 0.00000001))
                loss = cross_entropy 
                acc = tf.reduce_sum( feed_y * y * feed_r) 
                train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

                self._x = x; self._y = y; self._feed_y = feed_y; self._feed_r = feed_r
                self._loss = loss
                self._acc = acc
                self._train_step = train_step
	
	def nnstep(self, sess, cb, player, isMax = False):
		if player == PLAYER_BLACK:
			cb = cb.copy()
			cb = -cb
		startt = time.clock()
		nny = sess.run(self.y(), feed_dict={self.x(): cb.reshape([-1, BOARD_WIDTH, BOARD_WIDTH])})
		#print("nnstep %f" % (time.clock() - startt))
		#printpval(nny)
		p = np.argmax(nny)
		if not isMax:
			check_p = np.random.rand()
			sum_p = 0
			for i in range(len(nny[0])):
				sum_p += nny[0][i]
				if sum_p >= check_p:
					p = i
        				break
		position = (p // BOARD_WIDTH, p % BOARD_WIDTH)
		return position

def get_possible_pos(cb):
        pos_set = []
        delta = ((-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1), (0,-1))
        for i in range(BOARD_WIDTH):
                for j in range(BOARD_WIDTH):
                        if cb[i][j] == 0:
                                for d in range(len(delta)):
                                        pos = (i + delta[d][0],j + delta[d][1])
                                        if not isover(pos) and cb[pos[0]][pos[1]] != 0:
                                                pos_set.extend([[i, j]])
                                                break
        return pos_set

def check_win_pos(cb, player):
	pos_set = get_possible_pos(cb)
	while len(pos_set) > 0 :
		index_pos = np.random.random_integers(0, len(pos_set) - 1)
                pos = pos_set[index_pos]
                del(pos_set[index_pos])
		if cb[pos[0]][pos[1]] == 0:
			cb[pos[0]][pos[1]] = player
			w = win(cb, player, pos)
			cb[pos[0]][pos[1]] = 0
                        if w == 1:
				return pos		
	return (-1, -1)

def t(cb):
        n_dim = cb.ndim
        cb = np.reshape(cb.copy(), (BOARD_WIDTH, BOARD_WIDTH))
        t1 = np.zeros((BOARD_WIDTH, BOARD_WIDTH))
	t2 = np.zeros((BOARD_WIDTH, BOARD_WIDTH))
	t3 = np.zeros((BOARD_WIDTH, BOARD_WIDTH))
	t4 = np.zeros((BOARD_WIDTH, BOARD_WIDTH))
        for i in range(BOARD_WIDTH):
                for j in range(BOARD_WIDTH):
			t1[j][i] = cb[i][j]
			t2[BOARD_WIDTH - j - 1][BOARD_WIDTH - i - 1] = cb[i][j]
			t3[i][BOARD_WIDTH -j - 1] = cb[i][j]
			t4[BOARD_WIDTH - i - 1][j] = cb[i][j]
			
        if n_dim == 1:
                t1 = np.reshape(t1, (BOARD_AREA))
		t2 = np.reshape(t2, (BOARD_AREA))
		t3 = np.reshape(t3, (BOARD_AREA))
		t4 = np.reshape(t4, (BOARD_AREA))
		
        return (t1,t2,t3,t4)


def adds(cb, pos, r, sx, sy, sr):
	y_vector = np.zeros((BOARD_WIDTH, BOARD_WIDTH), dtype = float)
	y_vector[pos[0]][pos[1]] = 1
	sx.extend([cb.copy()])
	sy.extend([np.reshape(y_vector, [BOARD_AREA])])
	sr.extend([[r]])
	sx.extend(t(cb.copy()))
	sy.extend(t(np.reshape(y_vector, [BOARD_AREA])))
	sr.extend([[r], [r], [r], [r]])


def select_position(sess, cb, player, g):
	startt = time.clock()
        win_pos = check_win_pos(cb, player)
        if not isover(win_pos):
                return (1, win_pos)
	pos = g.nnstep(sess, cb, player)
	return (0, pos)

def updater(rs):
        l = len(rs)//5
	r = 1
	gama = 0.5
        for i in range(l):
		if r < 0.01:
			r = 0.01
                rs[(l - i) * 5 - 1][0] = r
		rs[(l - i) * 5 - 2][0] = r
		rs[(l - i) * 5 - 3][0] = r
		rs[(l - i) * 5 - 4][0] = r
		rs[(l - i) * 5 - 5][0] = r
		r = r * gama

def play(sess, g, round_count):
	xs = []; ys = []; rs = [];
	for i in range(round_count):
		#print("round_count %d" % i)
                sx1 = [];sy1 = [];sr1 = []
		sx2 = [];sy2 = [];sr2 = []

                cb = np.zeros((BOARD_WIDTH,BOARD_WIDTH), dtype=int)
                step_count = 0
		wplayer = 0
	
		current_player = PLAYER_WHITE
		if i % 2 == 0 or step_count !=1:
			current_player = PLAYER_BLACK
		cx = []; cy = []; cr = []	
                while step_count <= (BOARD_AREA - BOARD_AREA//5):
                        step_count += 1
			if current_player == PLAYER_WHITE:
				cx = sx1; cy = sy1; cr = sr1;
			else:
				cx = sx2; cy = sy2; cr = sr2;
				
                        (w, position) = select_position(sess, cb, current_player, g)
			adds(current_player * cb.copy(), position, 0, cx, cy, cr)

			cb[position[0]][position[1]] = current_player
			w =  win(cb, current_player, position)
			if w == 1:
				print("round_count %d step %d" % (i, np.sum(np.abs(cb))))
				#print(np.sum(np.abs(cb)))
				xs.extend(cx)
				ys.extend(cy)
				updater(cr)
				rs.extend(cr)
				#print(cr)
			if w != 0:
				break;
			current_player = -current_player
	return (xs, ys, rs)

def loadpart(path, index_id):
        xs = [];ys = [];rs = []
	print(path +'/xs.' + index_id)
        try:
                with open(path + '/xs.' + index_id, 'rb') as f:
                        xs = pickle.load(f)
                with open(path + '/ys.' + index_id, 'rb') as f:
                        ys = pickle.load(f)
                with open(path + '/rs.' + index_id, 'rb') as f:
                        rs = pickle.load(f)
        except (IOError) , e:
                print(e.message)
		return ([],[],[])

        return (xs, ys, rs)

def loadtestset(i=1):
        files= os.listdir("test")
        for f in files:
                (h, ext) = f.split('.')
		ss = ext.split('_')
                if h == "xs" and ss[len(ss) -1] == str(i):
			(x, y, r) = loadpart('test', ext)
			if len(x) > 100:
				print(len(x))
				return (x[0:100], y[0:100], r[0:100])
	return ([],[],[])

def loadsamples():
	xs=[]; ys=[]; rs=[];
	files= os.listdir("samples")
	exts = []
	for f in files:
		(h, ext) = f.split('.')
		if h == "xs":
			exts.extend([ext])
	
	while len(exts) > 0 :
		i = np.random.random_integers(0, len(exts) - 1)
		ext = exts[i]
		del(exts[i])
		(x, y, r) = loadpart('samples', ext)
                xs.extend(x)
                ys.extend(y)
                rs.extend(r)
	print("samples %d" % len(xs))
	x = [];	y = [];	r = []	
	l = len(xs)
	for i in range(100000):
		j = np.random.random_integers(0, l - 10 - 1)
		#if rs[j][0] > 0.1:
		x.extend(xs[j : j + 10])
		y.extend(ys[j : j + 10])
		r.extend(rs[j : j + 10])

	return (x, y, r)	
	
def main():
	g = mygraph()
	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = 0.15
	sess = tf.Session(config=config)	
	saver = tf.train.Saver()  
	sess.run(tf.initialize_all_variables())
	try:
		saver.restore(sess, "./m/go.ckpt")
		print("load model")
	except:
		print("no model")

	print("load testset")
	(tx1, ty1, tr1) = loadtestset(1)
	print("load samples ... ")
	(xs, ys, rs) = loadsamples()
	test_number = 100
        simplex_number = len(xs)
        mini_batch = 128
	print("load samples %d" % len(xs))
	tx = xs[simplex_number - test_number: simplex_number]	
	ty = ys[simplex_number - test_number: simplex_number]
	tr = rs[simplex_number - test_number: simplex_number]


	for i in range(50000):
		if i % 100 == 0:
			print("terain_step %d " % i)
			(l, a) = g.testloss(sess, tx, ty, tr)
			print("loss(%d, %d) = %.5f"  % (len(tx), a, l))
		
			if i % 500 == 0:
				print("save model")
				saver.save(sess, "./m/go.ckpt")
		
		start_position = i % (simplex_number // mini_batch) * mini_batch
		train_x = xs[start_position : start_position + mini_batch]
		train_y = ys[start_position : start_position + mini_batch]
		train_r = rs[start_position : start_position + mini_batch]
		sess.run(g.train(), feed_dict = {g.x() : train_x, g.feed_y() : train_y, g.feed_r() : train_r})
		
if __name__ == "__main__" :
    main()
