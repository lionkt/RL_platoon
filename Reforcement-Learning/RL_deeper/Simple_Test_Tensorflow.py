import tensorflow as tf
import numpy as np

MAX_STEP = 1000


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


def linear_official_test():
    # tensorflow 官方test
    # train data
    x_data = np.array([1, 2, 3, 4])
    y_data = np.array([0., -1., -2., -3.])
    # input and output
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    # model parameters
    weights = tf.Variable([.0], dtype=tf.float32)
    bias = tf.Variable([.0], dtype=tf.float32)
    linear_model = weights * x + bias
    # loss function
    loss = tf.reduce_mean(tf.square(y - linear_model))
    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    # train mainloop
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(MAX_STEP):
            sess.run(train, {x: x_data, y: y_data})

        # evaluate
        # final_W, final_b = sess.run([weights, bias])  # variable作为变量，存储了训练的结果。因此该语句可正常运行
        # final_loss = sess.run(loss) # 由于涉及placeholder的语句不直接存储数据，需要指定输入的数据
        final_W, final_b, final_loss = sess.run([weights, bias, loss], {x: x_data, y: y_data})
        print("w:%s, b:%s" % (final_W, final_b))






if __name__ == "__main__":
    # linear_regression()
    # test_placeholder()
    linear_official_test()
