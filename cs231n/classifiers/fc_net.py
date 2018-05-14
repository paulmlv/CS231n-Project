from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        W1 = np.random.normal(0.0, weight_scale, size = (input_dim, hidden_dim))
        W2 = np.random.normal(0.0, weight_scale, size = (hidden_dim, num_classes))
        b1 = np.zeros((hidden_dim,))
        b2 = np.zeros((num_classes,))
        self.params["W1"] = W1
        self.params["W2"] = W2
        self.params["b1"] = b1
        self.params["b2"] = b2
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        dim = 1
        for i in range(1, len(X.shape)):
            dim *= X.shape[i]
        X = np.reshape(X, (X.shape[0], dim))
        
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        reg = self.reg
        N, D = X.shape


        scores_step_1 = np.maximum(0, X.dot(W1) + b1)
        scores_step_2 = scores_step_1.dot(W2) + b2
        scores = scores_step_2
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        scores -= np.max(scores)
        exp_scores = np.exp(scores)
        normalization_matrix = np.ones(scores.shape) * np.transpose([np.sum(exp_scores, axis = 1)])
        proba_scores = exp_scores / normalization_matrix
    
        loss = np.sum(-np.log(proba_scores[np.arange(proba_scores.shape[0]), y]))
  
        num_train = X.shape[0]
        loss /= num_train

        # Add regularization to the loss.
        loss += 0.5 * reg * np.sum(W1 * W1)
        loss += 0.5 * reg * np.sum(W2 * W2)
        
        grads = {}
        X2 = scores_step_1
    
        proba_scores[np.arange(proba_scores.shape[0]),y] -= 1    
    
        dW2 = np.zeros_like(W2)
        dW2 = np.transpose(X2).dot(proba_scores)

        dW12 = proba_scores.dot(np.transpose(W2))
        dW12[scores_step_1 <= 0] = 0
        dW1 = np.transpose(X).dot(dW12)
    
        db2 = np.sum(proba_scores, axis = 0)
        db1 = np.sum(dW12, axis = 0)
    
        # Right now the loss is a sum over all training examples, but we want it
        # to be an average instead so we divide by num_train.
        num_train = X2.shape[0]
        dW2 /= num_train
        dW1 /= num_train
        db2 /= num_train
        db1 /= num_train

        # Add regularization to the gradient
        dW2 += reg * W2
        dW1 += reg * W1
        grads['W2'] = dW2
        grads['W1'] = dW1
        grads['b2'] = db2
        grads['b1'] = db1
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        L = []
        L.append(input_dim)
        for i in hidden_dims:
            L.append(i)
        L.append(num_classes)
        
        for i in range(1, self.num_layers):
            rows = L[i - 1]
            cols = L[i]
            Wi = np.random.normal(0.0, weight_scale, size = (rows, cols))
            bi = np.zeros((cols,))
            self.params["W" + str(i)] = Wi
            self.params["b" + str(i)] = bi
            
            if self.normalization == 'batchnorm':
                self.params["gamma" + str(i)] = np.ones((1, cols))
                self.params["beta" + str(i)] = np.zeros((1, cols))
            
            
        WL = np.random.normal(0.0, weight_scale, size = (L[-2], L[-1]))
        bL = np.zeros((L[-1],))
        self.params["W" + str(self.num_layers)] = WL
        self.params["b" + str(self.num_layers)] = bL
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        scores = X
        scores_dict, caches_dict, caches_dropout_dict = {}, {}, {}
        scores_dict[0] = scores
        
        for i in range(1, self.num_layers):
            Wi = self.params["W" + str(i)]
            bi = self.params["b" + str(i)] 
            
            if self.normalization == "batchnorm":
                gammai = self.params["gamma" + str(i)]
                betai = self.params["beta" + str(i)]
                
                scores, cache = affine_batch_relu_forward(scores, Wi, bi, gammai, betai, self.bn_params[i - 1])
            
            else:
                scores, cache = affine_relu_forward(scores, Wi, bi)
        
            if self.use_dropout:
                scores, cache_dropout = dropout_forward(scores, self.dropout_param)
                caches_dropout_dict[str(i)] = cache_dropout
            
            caches_dict[str(i)] = cache 
            scores_dict[str(i)] = scores
        
        
        WL = self.params["W" + str(self.num_layers)]
        bL = self.params["b" + str(self.num_layers)] 
        scores, cache = affine_forward(scores, WL, bL)
        scores_dict[str(self.num_layers)], caches_dict[str(self.num_layers)] = scores, cache
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        reg = self.reg
        loss, dout = softmax_loss(scores, y)
        
        cache = caches_dict[str(self.num_layers)]
        dout, dw, db = affine_backward(dout, cache)
        
        grads["W" + str(self.num_layers)] = dw + reg * self.params["W" + str(self.num_layers)]
        grads["b" + str(self.num_layers)] = db 
        loss += 0.5 * reg * np.sum(self.params["W" + str(self.num_layers)] ** 2)
        
        for i in range(self.num_layers - 1, 0, -1):
            cache = caches_dict[str(i)]
            
            if self.use_dropout:
                cache_dropout = caches_dropout_dict[str(i)]
                dout = dropout_backward(dout, cache_dropout)
            
            if self.normalization == "batchnorm":
                dout, dw, db, dgamma, dbeta = affine_batch_relu_backward(dout, cache)
                grads["gamma" + str(i)] = dgamma
                grads["beta" + str(i)] = dbeta
            
            else:
                dout, dw, db = affine_relu_backward(dout, cache)
                
                
            grads["W" + str(i)] = dw + reg * self.params["W" + str(i)]
            grads["b" + str(i)] = db 
            loss += 0.5 * reg * np.sum(self.params["W" + str(i)] ** 2)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
