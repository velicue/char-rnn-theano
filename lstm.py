import cPickle
import numpy as np
import theano
import theano.tensor as T
from theano import config
import random

import timeit, sys, os

def load_data():
    def shared_data(data, borrow=True):
        shared_ = theano.shared(np.asarray(data,
                                              dtype=theano.config.floatX),
                                borrow=borrow)
        return shared_

    word_index = np.load("feature.npy")

    return word_index.astype(np.int64)

def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(config.floatX)

class LSTMLayer(object):
    def __init__(self, word_emb_input, n_steps, n_samples, n_dim, n_out, n_levels, params = None):
        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n * dim:(n + 1) * dim]
            if _x.ndim == 1:
                return _x[n * dim:(n + 1) * dim]
            return _x[:, n * dim:(n + 1) * dim]

        def _step(x_, h_, c_, pred_, prob_):
            h_a = []
            c_a = []
            for it in range(self.n_levels):
                preact = T.dot(h_[it], self.U[it])
                preact += T.dot(x_, self.W[it]) + self.b[it]

                i = T.nnet.sigmoid(_slice(preact, 0, self.n_dim))
                f = T.nnet.sigmoid(_slice(preact, 1, self.n_dim))
                o = T.nnet.sigmoid(_slice(preact, 2, self.n_dim))
                c = T.tanh(_slice(preact, 3, self.n_dim))

                c = f * c_[it] + i * c
                h = o * T.tanh(c)

                h_a.append(h)
                c_a.append(c)

                x_ = h

            q = T.dot(h, self.L) + self.b0
            prob = T.nnet.softmax(q)
            pred = T.argmax(prob, axis=1)

            return T.stack(h_a).squeeze(), T.stack(c_a).squeeze(), pred, prob

        self.n_levels = n_levels
        self.n_dim = n_dim
        self.n_steps = n_steps
        self.n_samples = n_samples
        self.n_out = n_out
        self.slice = _slice

        if params == None:
            W = []
            U = []
            b = []
            for i in range(n_levels):
                W.append(np.concatenate([ortho_weight(n_dim),
                                       ortho_weight(n_dim),
                                       ortho_weight(n_dim),
                                       ortho_weight(n_dim)], axis=1))
                U.append(np.concatenate([ortho_weight(n_dim),
                                       ortho_weight(n_dim),
                                       ortho_weight(n_dim),
                                       ortho_weight(n_dim)], axis=1))
                b.append(np.zeros((4 * n_dim,)).astype(config.floatX))

            L = (np.zeros(
                (n_dim, n_out),
                dtype=theano.config.floatX
            ))
            b0 = np.zeros(n_out).astype(config.floatX)
            self.W = []
            self.U = []
            self.b = []
            for i in range(n_levels):
                self.W.append(theano.shared(
                    value=W[i],
                    name='lstm_W',
                    borrow=True
                ))

                self.U.append(theano.shared(
                    value=U[i],
                    name='lstm_U',
                    borrow=True
                ))

                self.b.append(theano.shared(
                    value=b[i],
                    name='lstm_b',
                    borrow=True
                ))

            self.L = (theano.shared(
                value=L,
                name='lstm_L',
                borrow=True
            ))
               

            self.b0 = theano.shared(
                value=b0,
                name='lstm_b0',
                borrow=True
            )

        else:
            self.W = params[:n_levels]
            self.U = params[n_levels:2 * n_levels]
            self.b = params[2 * n_levels:3 * n_levels]
            self.L = params[3 * n_levels]
            self.b0 = params[3 * n_levels + 1]

        rval, updates = theano.scan(_step,
                                    sequences=[word_emb_input],
                                    outputs_info=[T.alloc(np_floatX(0.),
                                                               n_levels,
                                                               n_samples,
                                                               n_dim),
                                                  T.alloc(np_floatX(0.),
                                                               n_levels,
                                                               n_samples,
                                                               n_dim),
                                                  T.alloc(np_int64(0),
                                                               n_samples),
                                                  T.alloc(np_floatX(0.),
                                                               n_samples,
                                                               n_out)],
                                    name="lstm_layers",
                                    n_steps=n_steps)

        self.output = rval[0]
        self.pred = rval[2]
        self.prob = rval[3]

        self.params = [self.W, self.U, self.b, [self.L], [self.b0]]

    def generate(self, h_, c_, x_):
        h_a = []
        c_a = []
        for it in range(self.n_levels):
            preact = T.dot(x_, self.W[it])
            preact += T.dot(h_[it], self.U[it]) + self.b[it]

            i = T.nnet.sigmoid(self.slice(preact, 0, self.n_dim))
            f = T.nnet.sigmoid(self.slice(preact, 1, self.n_dim))
            o = T.nnet.sigmoid(self.slice(preact, 2, self.n_dim))
            c = T.tanh(self.slice(preact, 3, self.n_dim))

            c = f * c_[it] + i * c
            h = o * T.tanh(c)

            h_a.append(h)
            c_a.append(c)

            x_ = h

        q = T.dot(h, self.L) + self.b0
        # mask = T.concatenate([T.alloc(np_floatX(1.), q.shape[0] - 1), T.alloc(np_floatX(0.), 1)])
        prob = T.nnet.softmax(q / 1)

        return prob, T.stack(h_a).squeeze(), T.stack(c_a)[0].squeeze()

    def negative_log_likelihood(self, y):
        a = np.array([np.arange(self.n_steps), ] * self.n_samples).T
        b = np.array([np.arange(self.n_samples), ] * self.n_steps)
        return -T.mean(T.log(self.prob)[a, b, y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            res = T.mean(T.neq(self.pred, y))
            return res
        else:
            raise NotImplementedError()

    def preds(self):
        return self.pred

    def probs(self):
        return self.prob


from theano import config
def np_floatX(data):
    return np.asarray(data, dtype=config.floatX)

def np_int64(data):
    return np.asarray(data, dtype=np.int64)

def adadelta_updates(parameters, gradients, rho, eps, scale):
    # create variables to store intermediate updates
    accugrads = [theano.shared(p.get_value() * np_floatX(0.)) for p in parameters]
    accudeltas = [theano.shared(p.get_value() * np_floatX(0.)) for p in parameters]

    # calculates the new "average" delta for the next iteration
    agrads = [rho * accugrad + (1 - rho) * g * g for accugrad, g in zip(accugrads, gradients) ]
 
    # calculates the step in direction. The square root is an approximation to getting the RMS for the average value
    dxs = [(T.sqrt(accudelta + eps) / T.sqrt(agrad + eps)) * g for accudelta, agrad, g in zip(accudeltas, agrads, gradients)]
 
    # calculates the new "average" deltas for the next step.
    accudeltas_new = [rho * accudelta + (1 - rho) * dx * dx for accudelta, dx in zip(accudeltas, dxs)]
 
    # Prepare it as a list f
    accugrads_updates = zip(accugrads, agrads)
    accudeltas_updates = zip(accudeltas, accudeltas_new)
    parameters_updates = [(p, p - d * scale) for p, d in zip(parameters, dxs) ]
    return accugrads_updates + accudeltas_updates + parameters_updates    

def training(n_dim=256, n_epochs=4000, batch_size=100, n_levels = 3, model = ""):
    import cPickle
    dictionary = cPickle.load(open("dictionary.pkl", "r"))
    params = None
    if model != "": params = cPickle.load(open(model, "rb"))

    feature = load_data()
    n_steps = feature.shape[0] - 1
    n_samples = feature.shape[1]
    n_batches = n_samples / batch_size

    n_train_batches = n_batches
    n_valid_batches = n_batches - n_train_batches

    print n_train_batches

    print '... building the model'

    n_words = feature.max() + 1

    print n_words

    if params == None:
        word_emb = theano.shared((0.01 * np.random.rand(n_words, n_dim)).astype(config.floatX), name = "word_emb", borrow = True)
    else:
        word_emb = params[0]
    feature_shared = T.cast(theano.shared(feature.astype(config.floatX), name = "feature_shared", borrow = True), 'int64')
    
    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # generate symbolic variables for input (x and y represent a
    # minibatch)
    bound = T.lscalar()
    word_emb_input_batch = word_emb[feature_shared[:, bound * batch_size:(bound + 1) * batch_size]]
    feature_shared_batch = feature_shared[:, bound * batch_size:(bound + 1) * batch_size]
    
    print "classifier"

    if params == None:
        classifier = LSTMLayer(
            word_emb_input = word_emb_input_batch[:-1],
            n_steps = n_steps,
            n_samples = batch_size,
            n_dim = n_dim,
            n_out = n_words,
            n_levels = n_levels
        )
    else:
        classifier = LSTMLayer(
            word_emb_input = word_emb_input_batch[:-1],
            n_steps = n_steps,
            n_samples = batch_size,
            n_dim = n_dim,
            n_out = n_words,
            n_levels = n_levels,
            params = params[1:]
        )

    print "model"

    h_T = T.fmatrix()
    c_T = T.fmatrix()
    x_T = T.fvector()
    pred_T = T.lscalar()
    generate = theano.function(
        inputs=[h_T, c_T, pred_T],
        outputs=classifier.generate(h_T, c_T, x_T),
        givens={
            x_T:word_emb[pred_T]
        }
    )

    cost = classifier.negative_log_likelihood(feature_shared_batch[1:])
    # theano.printing.debugprint(cost)
    params = [word_emb] + [x for sub in classifier.params for x in sub]
    gparams = [T.grad(cost, param) for param in params]
    updates = adadelta_updates(params, gparams, 0.95, 1 * 1e-6, 1)

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            bound: index
        }
    )
    # end-snippet-5

    print '... training'

    # early-stopping parameters
    patience = 200 * n_train_batches  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = 12
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    loss_final = []

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        total_cost = 0
        cPickle.dump(params, open("classifier_lstm_" + str(epoch) + ".pkl", "wb"), protocol=cPickle.HIGHEST_PROTOCOL)
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            print "Epoch:", epoch, "minibatch", minibatch_index, "cost:", minibatch_avg_cost

            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                pred = random.randint(0, n_words / 10)
                c = dictionary[pred]

                h_cul = np.zeros((n_levels, n_dim)).astype(config.floatX)
                c_cul = np.zeros((n_levels, n_dim)).astype(config.floatX)
                for i in range(2000):
                    prob, h_cul, c_cul = generate(h_cul, c_cul, pred)
                    t = random.random()
                    for j in range(prob.shape[1]):
                        t -= prob[0][j]
                        if t < 1e-8: break
                    pred = j
                    c += dictionary[pred]

                print c.encode("GBK", "ignore")

                
           
    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i') %
          (best_validation_loss * 100., best_iter + 1))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

if __name__ == '__main__':
    training()
