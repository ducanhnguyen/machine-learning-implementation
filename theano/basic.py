import theano
import theano.tensor as T


def dmatrix():
    x = T.dmatrix('x')
    y = T.dmatrix('y')
    z = x + y
    print(type(z.owner))
    print(z.type)


def dscalar():
    x = T.dscalar('x')
    y = T.dscalar('y')
    z = x + y
    sum = theano.function(inputs=[x, y], outputs=[z])
    print(sum(1, 2))


def scalar():
    x = T.scalar('x')
    y = T.scalar('y')
    z = x + y
    sum = theano.function(inputs=[x, y], outputs=[z])
    print(sum(1, 2))
    print(theano.pp(z))


def logistic_regression():
    x = T.scalar('x')
    z = 1 / (1 + T.exp(-1 * x))

    logistic = theano.function(inputs=[x], outputs=[z])
    print(logistic(2))


def multiple_formula():
    x = T.scalar('x')
    y = T.scalar('y')
    sum = x + y
    multiplication = x * y
    f = theano.function(inputs=[x, y], outputs=[sum, multiplication])
    print(f(1, 2))


def default_input():
    x = T.scalar('x')
    y = T.scalar('y')
    sum = x + y
    f = theano.function(inputs=[x, theano.In(y, value=1)], outputs=[sum])
    print(f(1, 2))
    print(f(1))


def shared_variable():
    x = T.scalar(name='x')
    y = T.scalar(name='y')
    sum = x + y
    count = theano.shared(value=0, name='count')  # is shared variable, use to count the number of calling sum
    count_update = count + 1
    f = theano.function(inputs=[x, y], outputs=[sum], updates=[(count, count_update)])

    print(count.get_value())
    f(1, 2)
    print(count.get_value())
    f(3, 4)
    print(count.get_value())


def random_number():
    from theano.tensor.shared_randomstreams import RandomStreams
    srng = RandomStreams(seed=234)
    rv_n = srng.normal(size=(2, 2))  # matrix 2x2 # normal distribution
    f = theano.function(inputs=[], outputs=[rv_n])
    print(f())


def array():
    import numpy as np
    Y = theano.shared(value=np.array([0.2, 0.3, 0.5]), name='Y')
    Y_hat = theano.shared(value=np.array([0.1, 0.2, 0.8]), name='Y_hat')
    cross_entropy = -(Y * T.log(Y_hat)).sum()
    f = theano.function(inputs=[], outputs=[cross_entropy])
    print(f())


def gradient_one_variable():
    x = T.scalar(name='x')
    y = T.scalar(name='y')

    cost = x ** 2
    grad = theano.gradient.grad(cost=cost, wrt=x)
    f = theano.function(inputs=[x], outputs=[grad])
    print(f(3))


def gradient_one_variable():
    x = T.scalar(name='x')
    y = T.scalar(name='y')

    cost = x ** 2
    grad = theano.gradient.grad(cost=cost, wrt=x)
    f = theano.function(inputs=[x], outputs=[grad])
    print(f(x=3))


def gradient_multiple_variables():
    x = T.scalar(name='x')
    y = T.scalar(name='y')

    cost = x ** 2 + y ** 2
    grad = theano.gradient.grad(cost=cost, wrt=[x, y])
    f = theano.function(inputs=[x, y], outputs=grad)  # no need [grad] because grad is an array
    print(f(x=3, y=1))


gradient_multiple_variables()
