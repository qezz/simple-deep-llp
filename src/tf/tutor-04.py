# rf.train

import tensorflow as tf

sess = tf.Session()

W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W*x + b

init = tf.global_variables_initializer()

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
# print("with loss function:")
# print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

# fixW = tf.assign(W, [-1.])
# fixb = tf.assign(b, [1.])
# sess.run([fixW, fixb])
# print("with loss (fixed weight values)")
# print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

sess.run(init) # reset values to incorrect defaults.
for i in range(1000):
    sess.run(train, {x: x_train, y: y_train})

print(sess.run([W, b]))

curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

