import numpy as np
import tensorflow as tf


def sum_scalar():
    x = tf.placeholder(dtype=tf.float32)
    y = tf.placeholder(dtype=tf.float32)
    z = x + y

    with tf.Session() as session:
        output = session.run(z, feed_dict={x: 2, y: 3})
        print(output)


def sum_matrix():
    x = tf.placeholder(dtype=tf.float32, shape=(2,))
    y = tf.placeholder(dtype=tf.float32, shape=(2,))
    z = x + y

    with tf.Session() as session:
        output = session.run(z, feed_dict={x: np.array([1, 2]), y: np.array([3, 4])})
        print(output)  # [4. 6.]


def sum_matrix_2():
    x = tf.placeholder(dtype=tf.float32, shape=(2,))
    y = tf.placeholder(dtype=tf.float32, shape=(2,))
    z = tf.math.add(x, y)  # the same as '+'

    with tf.Session() as session:
        output = session.run(z, feed_dict={x: np.array([1, 2]), y: np.array([3, 4])})
        print(output)  # [4. 6.]


def sum_matrix_3():
    x = tf.placeholder(dtype=tf.float32, shape=(4, 2))
    y = tf.placeholder(dtype=tf.float32, shape=(1, 2))
    z = tf.math.add(x, y)  # the same as '+'

    with tf.Session() as session:
        output = session.run(z, feed_dict={x: np.array([[1, 2], [3, 4], [5, 6], [7, 8]]),
                                           y: np.array([[3, 4]])})
        print(output)
        '''
        [[ 4.  6.]
         [ 6.  8.]
         [ 8. 10.]
         [10. 12.]]
        '''

def multiple():
    x = tf.placeholder(dtype=tf.float32, shape=(2, 2))
    y = tf.placeholder(dtype=tf.float32, shape=(2, 2))
    z = tf.matmul(x, y)

    with tf.Session() as session:
        output = session.run(z,
                             feed_dict={x: np.array([[1, 1], [2, 2]]),
                                        y: np.array([[3, 3], [4, 4]])
                                        })
        print(output)


def variable():
    # a variable is not a symbolic variable
    # a placeholder variable is a symbolic variable
    x = tf.Variable(dtype=tf.float32, initial_value=-1)
    y = tf.Variable(dtype=tf.float32, initial_value=-1)
    z = x + y

    with tf.Session() as session:
        output = session.run(z, feed_dict={x: 1, y: 2})
        print(output)


def variable_value():
    """
    use session.run(variable) to get the value of a variable
    :return:
    """
    threshold = tf.Variable(dtype=tf.int32, initial_value=10)
    init = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init)
        val = session.run(threshold)
        print(val)


def gradient_descent():
    x_train = np.array([3, 4])
    y_train = 4
    print('y = ' + str(y_train))

    # an observation
    x = tf.placeholder(dtype=tf.float32, shape=(2,))
    y = tf.placeholder(dtype=tf.float32)

    w = tf.Variable(dtype=tf.float32, initial_value=[-1, -1])
    b = tf.Variable(dtype=tf.float32, initial_value=-1)

    # tf.tensordot
    y_hat_computation = tf.tensordot(a=x, b=w, axes=1) + b
    cost = tf.square(y_hat_computation - y)
    train_op = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init)

        for i in range(100):
            session.run(train_op, feed_dict={x: x_train, y: y_train})
            print('w = ' + str(session.run(w)))
            print('b = ' + str(session.run(b)))

            error = session.run(cost, feed_dict={y: y_train, x: x_train})
            print('error = ' + str(error))

            print()


def sum():
    y = tf.placeholder(dtype=tf.float32, shape=(3,))
    y_hat = tf.placeholder(dtype=tf.float32, shape=(3,))

    tf_cost = tf.math.reduce_sum(y - y_hat)

    with tf.Session() as session:
        cost = session.run(tf_cost, feed_dict={y: [1, 2, 3], y_hat: [4, 5, 6]})
        print(cost)


def nn_non_type_casting():
    # x is array of floats. Therefore, all computations of x output float number
    x = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float32)
    mean, variance = tf.nn.moments(x=x, axes=[0], )  # mean and variance are float number

    with tf.Session() as sess:
        m, v = sess.run([mean, variance])
        print('mean = ' + str(m))
        print('variance = ' + str(v))


def nn_type_casting():
    # x is array of integers. Therefore, all computations of x output integer number
    x = tf.constant([[1, 2], [3, 4], [5, 6]])
    mean, variance = tf.nn.moments(x=x, axes=[0], )  # mean and variance are integer number

    with tf.Session() as sess:
        m, v = sess.run([mean, variance])
        print('mean = ' + str(m))
        print('variance = ' + str(v))


def bernoulli_sample_generation():
    import numpy as np
    rand = tf.random_uniform(shape=(10000, 1), seed=100)

    p_success = 0.5

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        uniform_values = session.run(rand)

        print('probability of success: ')
        print(p_success)

        print('uniform values:')
        print(uniform_values.flatten())

        # convert probability into 0 or 1
        bernoulli_values = []
        for item in uniform_values:
            if item > p_success:
                bernoulli_values.append(1)
            else:
                bernoulli_values.append(0)
        print('bernoulli values:')
        print(bernoulli_values)

        print('mean = ' + str(np.mean(bernoulli_values)))
        print('variance = ' + str(np.var(bernoulli_values)))

bernoulli_sample_generation()
