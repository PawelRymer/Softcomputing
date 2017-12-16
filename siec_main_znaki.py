import os
import theano
import cPickle
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
from unpickle import unpickle

srng = RandomStreams()


def floatX(X):
    return np.asarray(X, dtype=np.float64)


def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))


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


def RMSprop(cost, params, lr=0.0019, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates


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


def one_hot(x,n):
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x),n))
    o_h[np.arange(len(x)),x] = 1
    return o_h


def mnist(ntrain=60000,ntest=10000,onehot=True):

    fname = 'baza_uczaca_znaki.npy'
    trX = np.asarray(unpickle(fname, 28*28), np.uint8)
    fname = 'baza_uczaca_znaki_labels.npy'
    trY = np.asarray(unpickle(fname, 36), np.uint8)
    fname = 'baza_walidujaca_znaki.npy'
    teX = np.asarray(unpickle(fname, 28*28), np.uint8)
    fname = 'baza_walidujaca_znaki_labels.npy'
    teY = np.asarray(unpickle(fname, 36), np.uint8)


    randomize_training_set = np.arange(len(trX))
    randomize_test_set = np.arange(len(teX))
    np.random.shuffle(randomize_test_set)
    np.random.shuffle(randomize_training_set)

    trX = trX[randomize_training_set]
    trY = trY[randomize_training_set]
    teX = teX[randomize_test_set]
    teY = teY[randomize_test_set]

    trX = trX/255.
    teX = teX/255.

    trX = trX[:ntrain]
    trY = trY[:ntrain]

    teX = teX[:ntest]
    teY = teY[:ntest]

    return trX,teX,trY,teY

def chunks(l, n):
    for i in xrange(0, len(l), n):
        yield l[i : i + n]

def save_weights(weights, fname):
    with open(fname,"wb") as f:
        for w in weights:
            cPickle.dump(w, f)

if __name__ == "__main__":
    trX, teX, trY, teY = mnist(194909, 35813)

    trX = trX.reshape(-1, 1, 28, 28)
    teX = teX.reshape(-1, 1, 28, 28)

    X = T.tensor4(dtype='float64')
    Y = T.fmatrix()

    w = init_weights((4, 1, 3, 3))
    w2 = init_weights((10, 4, 3, 3))
    w3 = init_weights((20, 10, 3, 3))
    w4 = init_weights((20 * 2 * 2, 50))
    w_o = init_weights((50, 36))

    l1, l2, l3, l4, py_x = model(X, w, w2, w3, w4, 0., 0.)
    y_x = T.argmax(py_x, axis=1)

    cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y))
    params = [w, w2, w3, w4, w_o]
    updates = RMSprop(cost, params, lr=0.0019)

    train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
    predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

    print "start"
    best = 0.0
    minibatch = 50

    while True:
        err = 0.0
        indices = range(0, len(trX))
        np.random.shuffle(indices)

        for batch in chunks(indices, minibatch):
            err_cur = train(trX[batch], trY[batch])
            err += minibatch * err_cur

        err /= len(indices)
        cur = np.mean(np.argmax(teY, axis=1) == predict(teX))
        if cur > best:
            best = cur
            print
            print 'Zapisano nowy najlepszy wynik: ' + str(round(best,2)*100)+'%'
            save_weights([w, w2, w3, w4, w_o], 'wagi_4warstwy'+str(round(best,2)*100)+'%')
        #     save_weights([w, w2, w3, w4, w_o],'wagi')

        print "err, cur, best:", err, cur, best
        if cur > 0.95:
            print "dobry wynik, to juz to, dzieki za walke"
            exit()
