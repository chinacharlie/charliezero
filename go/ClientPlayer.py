# coding=utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


from abc import ABCMeta, abstractmethod

import urllib2
import gzip
import StringIO
import time


__userAgent = 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.4 (KHTML, like Gecko) Chrome/22.0.1229.79 Safari/537.4'


class __DefaultErrorHandler(urllib2.HTTPDefaultErrorHandler):

    def http_error_default(self, req, fp, code, msg, headers):
        result = urllib2.HTTPError(req.get_full_url(), code, msg, headers, fp)
        result.status = code
        return result


def _RequestWebPage(url, refer, lastmodified, etag, body, method, useragent, tmout=60):
    request = None
    try:
        request = urllib2.Request(url, data=body)
        if method:
            request.get_method = lambda: method
    except:
        return {'status': 600, 'data': None, 'err': '#600 request build failed:' + url}
    request.add_header('User-Agent', useragent)
    request.add_header(
        'Accept', 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8')
    request.add_header('Accept-encoding', 'gzip,sdch')
    request.add_header('Accept-Language', 'zh-CN,zh;q=0.8')
    request.add_header('Accept-Charset', 'gb2312;utf-8;q=0.7,*;q=0.3')
    if refer:
        request.add_header('Referer', refer)
    if lastmodified:
        request.add_header('If-Modified-since', lastmodified)
    if etag:
        request.add_header('If-None_match', etag)

    opener = urllib2.build_opener(__DefaultErrorHandler())
    try:
        try:
            f = opener.open(request, timeout=60)
        except:
            f = opener.open(request, timeout=60)

        if not hasattr(f, 'status'):
            f.status = 200
    except:
        return {'status': 601, 'data': None, 'err': '#601 open url error:' + url}

    try:
        data = f.read()
    except:
        return {'status': 602, 'data': None, 'err': '#602 read request data error:' + url}

    if not data:
        return {'status': 603, 'data': None, 'err': '#603 no data' + url}

    newlastmodified = 'last-modified' in f.headers and f.headers['last-modified']
    newetag = 'etag' in f.headers and f.headers['etag']
    expires = 'Expires' in f.headers and f.headers['Expires']

    if 'Content-Encoding' in f.headers and f.headers['Content-Encoding'] == 'gzip':
        compressedstream = StringIO.StringIO(data)
        try:
            gzipper = gzip.GzipFile(fileobj=compressedstream)
            data = gzipper.read()
        except:
            return {'status': 604, 'data': None, 'err': '#604 decompress data error:' + url}
    if 'Content-Type' in f.headers and ('gbk' in f.headers['Content-Type'] or 'gb2312' in f.headers['Content-Type']):
        try:
            data = data.decode(encoding='gbk', errors='ignore')
        except:
            return {'status': 605, 'data': None, 'err': '#605 decode gbk error' + url}
    if f.status != 200:
        data = None
    else:
        if 'Content-Length' in f.headers and len(data) != int(f.headers['Content-Length']):
            data = None
            f.status = 607
    return {'status': f.status, 'data': data, 'lastmodified': newlastmodified, 'etag': newetag, 'expires': expires, 'err': None}


def RequestWebPage(url, refer=None, lastmodified=None, etag=None, body=None, method=None, retrytimes=1, useragent=__userAgent, timeout=60):
    ret = {'status': 606, 'data': None, 'err': '#606'}
    for i in range(0, retrytimes + 1):
        ret = _RequestWebPage(url, refer, lastmodified,
                              etag, body, method, useragent, tmout=timeout)
        if ret['data']:
            break
        if ret['status'] >= 400 and ret['status'] < 500:
            break
    return ret


def RequestFile(url, savefile, refer=None, retrytimes=1):
    ret = RequestWebPage(url, refer=refer, retrytimes=retrytimes)
    if not ret['data'] or len(ret['data']) == 0:
        return False

    try:
        f = open(savefile, 'wb')
        try:
            f.write(ret['data'])
        except:
            return False
        finally:
            f.close()
    except:
        return False

    return len(ret['data'])


import copy

WIDTH = 15
AREA = WIDTH * WIDTH

SPACE = 0
WHITE = 1
BLACK = -1

NO_SPACE = 2


def cps(c):
    if c == BLACK:
        return 2
    return c


class Gobang(object):

    def reset(self):
        self._won = {}
        self._recent = []
        for j in xrange(WIDTH):
            for i in xrange(WIDTH):
                self._board[i][j] = SPACE

    def board(self):
        return self._board

    def get(self, row, col):
        if row < 0 or row >= WIDTH or col < 0 or col >= WIDTH:
            return 0
        return self._board[row][col]

    def put(self, row, col, x=None):
        if len(self._recent) == AREA:
            raise RuntimeError('put %d, %d board is full' % (row, col))
        if self._recent:
            last = self._recent[-1]
            if x is None:
                x = -last[2]
            if row < 0 or row >= WIDTH or col < 0 or col >= WIDTH:
                raise RuntimeError('put %d, %d out of boards' % (row, col))
            if x != -last[2]:
                raise RuntimeError('put %d, %d, %d color error' % (row, col, x))
        elif x is None:
            x = WHITE

        elif x != WHITE:
            raise RuntimeError('put %d, %d, %d must white first' % (row, col, x))
        if self._board[row][col] != SPACE:
            raise RuntimeError('put %d, %d, %d not empty' % (row, col, x))
        self._board[row][col] = x
        self._recent.append([row, col, x])

    def last_color(self):
        if self._recent:
            return self._recent[len(self._recent)-1][2]
        return 0

    def last(self):
        if self._recent:
            return self._recent[len(self._recent)-1]
        return None

    def recent(self):
        return self._recent

    def back(self, check=True):
        if not self._recent:
            raise RuntimeError('no step no back')
        last = self.last()
        self._board[last[0]][last[1]] = SPACE
        self._recent = self._recent[:-1]
        if check:
            self.check()
        return last

    def exchange(self, check=True):
        for j in xrange(WIDTH):
            for i in xrange(WIDTH):
                self._board[i][j] = -self._board[i][j]
        for j in xrange(len(self._recent)):
            self._recent[j][2] = -self._recent[j][2]
        if check:
            self.check()

    def step_num(self):
        return len(self._recent)

    def is_out(self, x, y):
        return x < 0 or y < 0 or x >= WIDTH or y >= WIDTH

    # 0（无输赢），WHITE（1 白棋赢），BLACK（-1 黑棋赢）， FULL_BOARD （2 无空格）
    def check(self):
        self._won = {}
        board = self._board
        dirs = ((1, -1), (1, 0), (1, 1), (0, 1))
        for i in xrange(WIDTH):
            for j in xrange(WIDTH):
                if board[i][j] == SPACE:
                    continue
                id = board[i][j]
                for d in dirs:
                    x, y = i, j
                    count = 0
                    for k in xrange(5):
                        if self.get(x, y) != id:
                            break
                        x += d[0]
                        y += d[1]
                        count += 1
                    if count == 5:
                        r, c = i, j
                        for z in xrange(5):
                            self._won[(r, c)] = 1
                            r += d[0]
                            c += d[1]
                        return id

        if len(self._recent) == AREA:
            return NO_SPACE
        return 0

    def dumps(self):
        import StringIO
        sio = StringIO.StringIO()
        board = self._board

        for i in self._recent:
            ti = chr(ord('A') + i[0])
            tj = chr(ord('A') + i[1])
            sio.write('%d%s%s' % (cps(i[2]), ti, tj))
        return sio.getvalue()

    def loads(self, text, check=True):
        self.reset()
        board = self._board
        s = text
        if text[1] == ':':
            s = text.replace(':', '')
        if len(s) % 3 != 0:
            raise RuntimeError('load failed, char num error: %d' % len(s))
        if not s:
            return

        n = len(s.split('1')) - len(s.split('2'))
        if n > 1 or n < -1:
            raise RuntimeError('load failed, white black num error')

        for k in xrange(0, len(s), 3):
            stone = int(s[k])
            if stone == 2:
                stone = -1
            i = ord(s[k + 1].upper()) - ord('A')
            j = ord(s[k + 2].upper()) - ord('A')
            board[i][j] = stone
            self._recent.append([i, j, stone])
        if check:
            self.check()

    # 彩色输出
    def show(self):
        print '   ' + ' 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5'[:WIDTH * 2]
        print '   ' + ' A B C D E F G H I J K L M N O P Q R S T U V W X Y Z'[:WIDTH * 2]
        mark = ('. ', 'O ', 'X ')
        nrow = 0
        self.check()
        color1 = 10
        color2 = 13
        last = (-1, -1)
        if self._recent:
            last = self._recent[len(self._recent)-1]
        for row in xrange(WIDTH):
            print row % 10, chr(ord('A') + row),
            for col in xrange(WIDTH):
                ch = self._board[row][col]
                if ch == SPACE:
                    self.console(-1)
                    print '.',
                elif ch == WHITE:
                    if row == last[0] and col == last[1]:
                        self.console(12)
                    elif (row, col) in self._won:
                        self.console(9)
                    else:
                        self.console(10)
                    print 'O',
                elif ch == BLACK:
                    if row == last[0] and col == last[1]:
                        self.console(12)
                    elif (row, col) in self._won:
                        self.console(9)
                    else:
                        self.console(13)
                    print 'X',
            self.console(-1)
            print row % 10, chr(ord('A') + row),
            print ''
        print '   ' + ' A B C D E F G H I J K L M N O P Q R S T U V W X Y Z'[:WIDTH * 2]
        print '   ' + ' 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5'[:WIDTH * 2]

    def __init__(self, cp=None):
        self._won = {}
        self._recent = []
        if cp is not None:
            self._board = copy.deepcopy(cp._board)
            self._recent = copy.deepcopy(cp._recent)
            self.check()
        else:
            self._board = [[0] * WIDTH for row in range(WIDTH)]

    def __getitem__(self, row):
        return self._board[row]

    def __str__(self):
        text = ' ' + ' A B C D E F G H I J K L M N O P Q R S T U V W X Y Z'[:WIDTH*2] + '\n'
        mark = ('. ', 'O ', 'X ')
        nrow = 0
        for row in self._board:
            line = ''.join([mark[cps(n)] for n in row])
            text += chr(ord('A') + nrow) + ' ' + line
            nrow += 1
            if nrow < WIDTH:
                text += '\n'
        return text

    def __repr__(self):
        return self.__str__()

    # 设置终端颜色
    def console(self, color):
        if sys.platform[:3] == 'win':
            try:
                import ctypes
            except:
                return 0
            kernel32 = ctypes.windll.LoadLibrary('kernel32.dll')
            GetStdHandle = kernel32.GetStdHandle
            SetConsoleTextAttribute = kernel32.SetConsoleTextAttribute
            GetStdHandle.argtypes = [ctypes.c_uint32]
            GetStdHandle.restype = ctypes.c_size_t
            SetConsoleTextAttribute.argtypes = [ctypes.c_size_t, ctypes.c_uint16]
            SetConsoleTextAttribute.restype = ctypes.c_long
            handle = GetStdHandle(0xfffffff5)
            if color < 0:
                color = 7
            result = 0
            if (color & 1):
                result |= 4
            if (color & 2):
                result |= 2
            if (color & 4):
                result |= 1
            if (color & 8):
                result |= 8
            if (color & 16):
                result |= 64
            if (color & 32):
                result |= 32
            if (color & 64):
                result |= 16
            if (color & 128):
                result |= 128
            SetConsoleTextAttribute(handle, result)
        else:
            if color >= 0:
                foreground = color & 7
                background = (color >> 4) & 7
                bold = color & 8
                sys.stdout.write(" \033[%s3%d;4%dm" % (bold and "01;" or "", foreground, background))
                sys.stdout.flush()
            else:
                sys.stdout.write(" \033[0m")
                sys.stdout.flush()
        return 0


def pos2char(row, col):
    return (chr(ord('A') + row), chr(ord('A') + col))


def run():
    ip = '119.29.118.24'
    if len(sys.argv) == 2:
        ip = sys.argv[1]
    go = Gobang()
    go.show()
    while go.check() == 0:
        print 'your color: ', 'WHITE (O)'
        x, y = _input()
        while x < 0 or y < 0 or x >= WIDTH or y >= WIDTH or go.get(x, y) != SPACE:
            if go.get(x, y) != SPACE:
                print 'position', x, y, 'not space!'
            else:
                print 'input error, retry!'
            x, y = _input()
        go.put(x, y)
        go.show()
        if go.check() == WHITE:
            print 'you win!'
            break
        print 'Server turn, Please Wait...'
        s = go.dumps()
        r = send(ip, s)
        if len(r) == 0:
            sys.exit(-2)

        go.loads(r)
        go.show()
        if go.check() == BLACK:
            print 'you lose!'
            break
        last = go.last()
        print 'NN player go', pos2char(last[0], last[1]), '\n'

def _input():
    while True:
        print 'type "quit" for x to end game'
        x = raw_input("x(0-15):")
        if x == 'quit':
            print 'exit'
            sys.exit(0)
        y = raw_input("y(0-15):")
        if len(x) == 0 or len(y) == 0:
            print 'input error, retry!'
            continue
        try:
            return int(x), int(y)
        except:
            print 'input error, retry!'
            continue

def send(ip, s):
    url = 'http://' + ip + ':8810/?board=' + s
    data = RequestWebPage(url, retrytimes=3, timeout=120)
    if data['status'] != 200:
        print data
        print 'service error!'
        return ''
    return data['data']


def test():
    g = Gobang()
    g.loads('2HH1II2IH1JH2GH1HJ2KG1FH2HG1GI2IK1JI2FI1IF2JG1IG2HI1GJ2IJ1JK2JJ1HF2KI1LH2GF1IE2KJ1KF2JF1GG')
    print g
    g.show()

    g1 = Gobang(g)

    g.put(0, 0)
    g.show()

    for _ in xrange(10):
        g.back()
    g.show()

    print '\ndumps:', g.dumps()
    print '\nrecent:', g.recent()
    print '\nlast:', g.last()
    print '\nlast_color:', g.last_color()
    print '\nstep_num:', g.step_num()
    print '\nget:', g.get(8, 8)
    print '\nreset'
    g.reset()
    g.show()

    ip = '119.29.118.24'
    print 'request...'
    ret = send(ip, g1.dumps())
    print 'server step:', ret
    g.loads(ret)
    g.show()


if __name__ == '__main__':
    run()
    #test()
