import tensorflow as tf
import numpy as np
import os
dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
datay_max = np.load(dir+"/model/datay_max.npy")
datay_min = np.load(dir+"/model/datay_min.npy")
def prediction(X):
    X = np.array(X)
    X = X.reshape([1,11])
    sess1 = tf.Session()
    saver = tf.train.import_meta_graph(dir+"/model/factory.ckpt.meta")
    ckpt = tf.train.get_checkpoint_state(dir+"/model")
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess1, ckpt.model_checkpoint_path)
    graph = tf.get_default_graph()
    x = graph.get_operation_by_name('x').outputs[0]
    pred = tf.get_collection("pred")[0]
    Y = sess1.run(pred, feed_dict={x: X})
    Y = Y*(datay_max-datay_min+0.0001)+datay_min
    print(Y)
    return Y.reshape(37)