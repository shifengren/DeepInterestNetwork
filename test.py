
import tensorflow as tf
import numpy as np

mt_idx = np.asarray([
    [1, 0, 2, 1],
    [3, 2, 0, 1]
])

mat = np.arange(8).reshape((2,4))
arry = np.arange(8)


mat = tf.convert_to_tensor(mat, dtype=tf.int64)
mt_idx = tf.convert_to_tensor(mt_idx, dtype=tf.int64)

sidx = tf.convert_to_tensor([0], dtype=tf.int64)


ret = tf.gather(arry, sidx)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sidx.eval())
    ret_out = sess.run([ret])
    print("ret_out:", ret_out)
    print("ret_out.shape", len(ret_out))