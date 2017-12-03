import cPickle
from PIL import Image
import numpy as np
import theano
from theano import tensor as T
from theano.tensor.nnet.conv import conv2d
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.signal.downsample import max_pool_2d
from math import fabs
import sys

from unpickle import unpickle

srng = RandomStreams()


def load_weights(fname):
    all = []
    with open(fname, "r") as f:
        while True:
            try:
                im = cPickle.load(f)
                all.append(im)
            except EOFError:
                print("Loaded!")
                break
    return all


def wczytaj_obrazek_szary(fname):
    kol = Image.open(fname)
    img = Image.open(fname).convert('L')
    height = int(img.size[1])
    width = int(img.size[0])

    difference = int(fabs((height - width) / 2))

    if height >= width:
        area = (0, difference, width, difference + width)
        new_img = img.crop(area)
        new_kol = kol.crop(area)
    else:
        area = (difference, 0, difference + height, height)
        new_img = img.crop(area)
        new_kol = kol.crop(area)

    sz = new_img.size[0]
    new_img = new_img.resize((sz,sz))
    new_kol = new_kol.resize((sz,sz))
    name = 'kwadrat' + fname
    new_kol.save(name)
    return new_img, name


def wczytaj_obrazek(fname, new_size):

    img = Image.open(fname).convert('L')

    height = int(img.size[1])
    width = int(img.size[0])

    difference = int(fabs((height - width) / 2))

    if height >= width:
        area = (0, difference, width, difference + width)
        new_img = img.crop(area)
        new_img = new_img.resize((new_size, new_size))
    else:
        area = (difference, 0, difference + height, height)
        new_img = img.crop(area)
        new_img = new_img.resize((new_size, new_size))

    new_img = np.asarray(new_img, np.float64)

    return new_img.reshape(-1, 1, new_size, new_size), height, width


def resize_image(img, size):
    new_img = img.resize((size,size))
    return np.array(new_img).reshape(-1,1,size,size)


def model(X, w, w2, w3, w4, p_drop_conv, p_drop_hidden):
    l1a = rectify(conv2d(X, w, border_mode='full'))
    l1 = max_pool_2d(l1a, (2, 2))
    l1 = dropout(l1, p_drop_conv)

    l2a = rectify(conv2d(l1, w2))
    l2 = max_pool_2d(l2a, (2, 2))
    l2 = dropout(l2, p_drop_conv)

    l3a = rectify(conv2d(l2, w3))
    l3b = max_pool_2d(l3a, (3, 3))
    l3 = T.flatten(l3b, outdim=2)
    l3 = dropout(l3, p_drop_conv)

    l4 = rectify(T.dot(l3, w4))
    l4 = dropout(l4, p_drop_hidden)

    pyx = softmax(T.dot(l4, w_o))
    return l1, l2, l3, l4, pyx


def rectify(X):
    return T.maximum(X, 0.)


def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')


def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype='float64')
        X /= retain_prob
    return X


def load_test_pictures(data, mode=0, data_labels=None):

    testY = None
    testX = None

    if mode == 1 and data_labels is not None:
        testX = np.load(data)
        testY = np.load(data_labels)
        testX = testX.reshape(-1, 1, 28, 28)
        testX = testX / 255.
    elif mode == 2 and data_labels is not None:
        testX = unpickle(data, 28*28)
        testY = unpickle(data_labels, 7)
        testX = testX.reshape(-1, 1, 28, 28)
        testX = testX / 255.
    elif mode == 3:
        img, width, height = wczytaj_obrazek(data, 28)
        img[0] = img[0] / 255.
        testX = img
        testY = np.zeros
    else:
        print 'Wrong open mode!'
        return testX, testY

    return testX, testY


if __name__ == "__main__":

    CODE = {'a': '.-', 'b': '-...', 'c': '-.-.',
            'd': '-..', 'e': '.', 'f': '..-.',
            'g': '--.', 'h': '....', 'i': '..',
            'j': '.---', 'k': '-.-', 'l': '.-..',
            'm': '--', 'n': '-.', 'o': '---',
            'p': '.--.', 'q': '--.-', 'r': '.-.',
            's': '...', 't': '-', 'u': '..-',
            'v': '...-', 'w': '.--', 'x': '-..-',
            'y': '-.--', 'z': '--..',

            '0': '-----', '1': '.----', '2': '..---',
            '3': '...--', '4': '....-', '5': '.....',
            '6': '-....', '7': '--...', '8': '---..',
            '9': '----.'
            }

    output = {
        0: '0', 1: '1', 2: '2',
        3: '3', 4: '4', 5: '5',
        6: '6', 7: '7', 8: '8',
        9: '9',

        10: 'a', 11: 'b', 12: 'c',
        13: 'd', 14: 'e', 15: 'f',
        16: 'g', 17: 'h', 18: 'i',
        19: 'j', 20: 'k', 21: 'l',
        22: 'm', 23: 'n', 24: 'o',
        25: 'p', 26: 'q', 27: 'r',
        28: 's', 29: 't', 30: 'u',
        31: 'v', 32: 'w', 33: 'x',
        34: 'y', 35: 'z'

    }

    opening_mode = {'data_file': 1,
                    'pickled_file': 2,
                    'image': 3}

    # data_amount = sys.argv[1]
    weights_file = sys.argv[1]  # file with weights values
    test_set = sys.argv[2]  # file(s) with picture(S)
    mode = sys.argv[3]  # 1 - read data set straight from pixel values, 2 - read pickled file, 3 - read image
    if len(sys.argv) > 4:
        test_set_labels = sys.argv[4]  # optional - file with label(s) to compare with the result
    else :
        test_set_labels = None

    teX, teY = load_test_pictures(test_set, opening_mode[mode], test_set_labels)
    if teX is None or teY is None:
        print 'Error while loading test pictures!'
        exit()

    [w, w2, w3, w4, w_o] = load_weights(weights_file)
    X = T.tensor4(dtype='float64')
    l1, l2, l3, l4, py_x = model(X, w, w2, w3, w4, 0., 0.)
    y_x = T.argmax(py_x, axis=1)
    predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

    # img = wczytaj_obrazek(im_name, 28)
    # img_np = np.array(img).reshape(-1,1,28,28)
    # img_pixels = img[0]/255.
    # pixels_values = np.asarray(img[0], np.float64)

    wynik = predict(teX)
    if opening_mode[mode] != 3:
        cur = np.mean(np.argmax(teY, axis=1) == wynik) * 100
        print 'Skutecznosc klasyfikacji sieci: %d%%' % cur
    else:
        print('Recognized character: ' + output[wynik[0]])
        print('Morse code of recognized character: ' + CODE[output[wynik[0]]])
