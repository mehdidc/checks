from theano import function, config, shared, sandbox
import theano.tensor as T
import numpy
import time

iters = 1000

size_1 = (1000, 28*28)
size_2 = (28*28, 500)

rng = numpy.random.RandomState(22)

m1 = shared(numpy.asarray(rng.normal(size=size_1), config.floatX))
m2 = shared(numpy.asarray(rng.normal(size=size_2), config.floatX))
bias = shared(numpy.asarray(rng.normal(size=(size_2[1],) ), config.floatX))


if config.device.startswith('gpu'):
	F = sandbox.cuda.basic_ops.gpu_from_host
else:
	F = lambda x:x

f = function([], F(T.tanh(T.dot(m1, m2)+bias)) )
print f.maker.fgraph.toposort()
t0 = time.time()
for i in xrange(iters):
    r = f()
t1 = time.time()
print 'Looping %d times took' % iters, t1 - t0, 'seconds'
print 'Result is', numpy.asarray(r)
if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
    print 'Used the cpu'
else:
    print 'Used the gpu'
