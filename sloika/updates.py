from collections import OrderedDict
import numpy as np
import theano as th
import theano.tensor as T

def sgd(network, loss, rate, momentum, clip=0.1):
    """  Stochastic Gradient Descent with momentum

    :param network: network to optimise
    :param loss: loss function to optimise over
    :param rate: rate (step size) for SGD
    :param momentum: momentum (decay for previous steps)

    :returns: a dictionary containing update functions for Tensors
    """
    assert momentum >= 0, "Momentum for SGD must be non-negative"

    params = network.params()
    updates = OrderedDict()
    gradients = th.grad(loss, params)
    for param, grad in zip(params, gradients):
        val = param.get_value(borrow=True)
        vel = th.shared(np.zeros(val.shape, dtype=val.dtype))
        updates[vel] = momentum * vel - rate * grad
        updates[param] = param + T.clip(updates[vel], -clip, clip)

    return updates

 def adam(network, loss, rate, decay, epsilon=1e-8):                              
     """  ADAM optimiser                                                          
                                                                                  
     :param network: network to optimise                                          
     :param loss: loss function to optimise over                                  
     :param rate: rate (step size) for optimiser                                  
     :param decay: decay for estimate of gradient and curvature                   
     :param epsilon: same parameter to prevent reciprocal of variance exploding   
                                                                                  
     :returns: a dictionary containing update functions for Tensors               
     """                                                                          
     assert decay >= (0.0, 0.0), "Decay must be non-negative"                     
     assert decay <= (1.0, 1.0), "Decay must be less-than or equal to one"        
                                                                                  
     params = network.params()                                                    
     updates = OrderedDict()                                                      
     gradients = th.grad(loss, params)                                            
                                                                                  
     ldecay = np.log(decay)                                                       
                                                                                  
     t = T.scalar(0)                                                              
     lr_t = T.scalar(rate)                                                        
     updates[t] = t + 1.0                                                         
     updates[lr_t] = rate * T.sqrt(-expm1(t * ldecay[0])) / -T.expm1(t * ldecay[1])
     for param, grad in zip(params, gradients):                                   
         val = param_in.get_value(borrow=True)                                    
                                                                                  
         momentum = th.shared(np.zeros(val.shape, dtype=val.dtype))               
         variance = th.shared(np.zeros(val.shape, dtype=val.dtype))               
         updates[momentum] = decay[0] * momentum + (1.0 - decay[0]) * grad        
         updates[variance] = decay[1] * variance + (1.0 - decay[1]) * T.sqr(grad) 
         update[param] = param - lt_t * momentum / (T.sqrt(variance) + epsilon)   

     return updates  


def edam(network, loss, rate, decay, epsilon=1e-4, clip=0.1):
    """  Exponential Decay Adaptive Momentum
    (similar to ADAM optimiser)

    :param network: network to optimise
    :param loss: loss function to optimise over
    :param rate: rate (step size) for SGD
    :param decay: decay for estimate of gradient and curvature


    :returns: a dictionary containing update functions for Tensors
    """
    assert decay >= (0.0, 0.0), "Decay must be non-negative"
    assert decay <= (1.0, 1.0), "Decay must be less-than or equal to one"


    params = network.params()
    updates = OrderedDict()
    gradients = th.grad(loss, params)
    for param, grad in zip(params, gradients):
        val = param.get_value(borrow=True)
        gr = th.shared(np.zeros(val.shape, dtype=val.dtype))
        cu = th.shared(np.ones(val.shape, dtype=val.dtype))
        n0 = th.shared(np.float32(0.0).astype(val.dtype))
        n1 = th.shared(np.float32(1.0).astype(val.dtype))

        updates[gr] = decay[0] * gr + grad
        updates[n0] = 1.0 + decay[0] * n0
        step = (updates[gr] / updates[n0]) / T.sqrt((cu + epsilon) / (epsilon + n1))
        updates[param] = param - T.clip(rate * step, -clip, clip)
        updates[cu] = decay[1] * cu + T.square(grad)
        updates[n1] = 1.0 + decay[1] * n1

    return updates
