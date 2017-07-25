import tensorflow as tf
import numpy as np

MAX_STEP = 200


def linear_regression():
    # 代码段功能学习线性拟合的参数
    # 构造数据集
    x_data = np.random.rand(100).astype(np.float32)
    y_data = x_data * 0.1 + 0.3

    weight = tf.Variable(tf.random_uniform([1], -1, 1))
    # bias = tf.Variable(tf.zeros([1]))
    bias = tf.Variable(0.0)
    y = weight * x_data + bias

    loss = tf.reduce_mean(tf.square(y - y_data))  # 这里只是定义运算，而不是执行
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.4)  # 采用gradient-descent法优化
    train = optimizer.minimize(loss)
    init = tf.global_variables_initializer()  # 初始化对于variable的使用是必要的步骤

    # 训练的过程
    # 对于tf.Session()尽量用with语句块，这样能保证出了with块session就自动关闭。否则一直在内存里
    with tf.Session() as sess:
        sess.run(init)
        for i in range(MAX_STEP):
            sess.run(train)
            if i % 20 == 0:
                print('step: ', i, ', weight: ', sess.run(weight), ', bias: ', sess.run(bias))  # 这里必须要加sess.run()才能访问结果


def test_placeholder():
    # 在 Tensorflow 中需要定义 placeholder 的 type ，一般为 float32 形式
    input1 = tf.placeholder(tf.float32)
    input2 = tf.placeholder(tf.float32)
    output = tf.multiply(input1, input2)
    with tf.Session() as sess:
        res = sess.run(output, feed_dict={input1: [7.], input2: [2.]})
        print(res)


if __name__ == "__main__":
    # linear_regression()
    test_placeholder()
