# coding=utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

from BaseHTTPServer import HTTPServer, BaseHTTPRequestHandler
import thread
import threading
import time
import tempfile
import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import f
import ClientPlayer



class OPlayerHTTPHandler(BaseHTTPRequestHandler):
    _threadLock = threading.Lock()

    @staticmethod
    def lock():
        OPlayerHTTPHandler._threadLock.acquire()

    @staticmethod
    def unlock():
        OPlayerHTTPHandler._threadLock.release()

    def __init__(self, request, client_address, server):
        BaseHTTPRequestHandler.__init__(self, request, client_address, server)

    def do_GET(self):
        parts = self.path.split('?')
        if len(parts) != 2:
            return
        if parts[1][:6] != 'board=':
            return
        s = parts[1][6:].strip()

        OPlayerHTTPHandler.lock()
        ret = OnlineNNPlayer.instance.new_board(s)
        OPlayerHTTPHandler.unlock()
        time.sleep(0.5)  # 避免多线程print显示混乱

        self.send_response(200)
        self.end_headers()
        self.wfile.write(ret)


class OnlineNNPlayer(object):
    instance = None

    def __init__(self, delegate):
        OnlineNNPlayer.instance = self
        self.delegate = delegate
        self._http_server = HTTPServer(
            ('', 20001), OPlayerHTTPHandler)

    def new_board(self, s):
        return self.delegate(s)

    def listen_and_serve(self):
        self._http_server.serve_forever()

g = f.mygraph()
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


def Charlie(s):
    print(s)
    go = ClientPlayer.Gobang()
    go.reset()
    go.loads(s)
    current_player = f.PLAYER_WHITE
    if np.sum(go.board()) > 0:
	current_player = f.PLAYER_BLACK
    if np.sum(go.board()) < 0:
        current_player = f.PLAYER_WHITE

    if np.sum(go.board()) == 0:
        if s[0] == '1':
            current_player = f.PLAYER_BLACK
        else:
            current_player = f.PLAYER_WHITE

    cb = np.zeros((f.BOARD_WIDTH,f.BOARD_WIDTH), dtype=int)
    for m in range(f.BOARD_WIDTH):
        for n in range(f.BOARD_WIDTH):
            cb[m][n] = go.board()[m][n]
    position = g.nnstep(sess, cb, current_player, True)
    go.put(position[0], position[1])
    go.show()
    return go.dumps() 


if __name__ == '__main__':
    print("listen")
    OnlineNNPlayer(Charlie).listen_and_serve()
