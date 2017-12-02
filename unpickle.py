import cPickle
import numpy as np

def unpickle(fname, size):
    all = []
    with open(fname) as f:
        while True:
            try:
                im = cPickle.load(f)
                if type(im) is np.ndarray:
                    im = im.reshape(size)
                all.append(im)
            except EOFError:
                print("Loaded!")
                break
    all = np.asarray(all)
    return all


