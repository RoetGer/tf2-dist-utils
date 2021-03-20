import tensorflow as tf

from tf2_dist_utils.distributions import TransNormal


class NegLogLikeLoss(tf.keras.losses.Loss):
    '''Base class for negative loglikelihood based losses'''

    def call(self, y_true, y_pred):
        '''Computes log-probability of observations
        
        Parameters
        ----------
        y_true : tf.tensor
            Observations
        y_pred : tf.tensor
            Parameter values of the distribution underlying the loss.
        
        Returns
        -------
        tf.tensor
            Negative loglikelihood for each observation.
        '''

        shape = y_pred.shape
        
        if len(y_pred.shape) > 1:
          lst = tf.split(
            y_pred, 
            num_or_size_splits=shape[-1], 
            axis=(shape.ndims - 1))
        else:
          lst = [y_pred]
        
        rv = self.dist(*lst)

        return -rv.log_prob(y_true)


def build_loss(class_loss_name, dist, params=None, **kwargs):
    '''Creates a tensorflow 2.x loss function based on NegLokLikeLoss

    Parameters
    ----------
    class_loss_name : str
        Name/Type of the loss to be created, e.g. "TransNormal"
    dist : distribution object
        Can be any type of tensorflow distribution object, 
        which supports parameter specification via call 
        (e.g. dist(param1, param2)) and has a log_prob method
    params : list[str]
        Specifies which parameters should be exposed for optimization
    **kwargs : dict
        kwargs can be used to fix attributes of the distribution
        object, e.g. if you want to create a loss function based on a
        Gaussian distribution with location set to 0, you can simply
        forward loc=0.0.

    Returns
    -------
    Object of type class_loss_name
        Loss object of type class_loss_name which can be used
        in conjuction with tensorflow 2.x models.
    '''
    
    if kwargs:
        dist = partial(dist, **kwargs)

    if params:
        dist = expose_params(dist, params)

    def __init__(self):
        self.dist = dist
        super(NegLogLikeLoss, self).__init__()

    loss = type(
        class_loss_name,
        (NegLogLikeLoss,),
        {
            "__init__": __init__
        }
    )

    return loss


def expose_params(func, params):
    '''Expose subset of parameters for optimization
    
    Useful if you are in a situation where only a subset of attributes
    of a function is relevant and you want them to be optimized

    Parameters
    ----------
    func : function/object
        Can be any type of function or object
    params : list[str]
        Specifies which parameters should be exposed for optimization
    Returns
    -------
    Object of type ExpParam
        This class call method has only attributes defined
        in params.
    '''

    func_header = ", ".join(params)
    func_call = ", ".join([
        p + "=" + p for p in params
    ])

    nparams = len(params)

    func_str = f'''
    class ExpParam:
      def __init__(self, func):
        self.func = func
        self.nparams = {nparams}
      
      def __call__(self, {func_header}):
          return self.func({func_call})
      
      def get_nparams(self):
          return self.nparams
    '''

    exec(inspect.cleandoc(func_str))

    return eval("ExpParam(func)")


NegGaussLogLikeLoss = build_loss("NegGaussLogLikeLoss", TransNormal)