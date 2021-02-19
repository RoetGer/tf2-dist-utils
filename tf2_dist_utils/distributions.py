from functools import wraps

import tensorflow_probability as tfp
tfd = tfp.distributions

def transform_param(cls, **transf_dics):
    
    @wraps(cls)
    def wrapped_class_init_(*args, **kwargs):
        varnames = cls.__init__.__code__.co_varnames[1:]
        args = list(args)
        len_args = len(args)

        for var, trans in transf_dics.items():
            idx = varnames.index(var)

            if idx < len_args:
                args[idx] = trans(args[idx])
            else:
                kwargs[var] = trans(kwargs[var])
        
        return cls(*args, **kwargs)
    
    return wrapped_class_init_


def build_zero_infl_dist(dist):
    '''Creates a zero-inflated distribution

    Parameters
    ----------
    dist : tfp.distribution object
        Base distribution used for creating the zero-inflated
        distribution, i.e. the distribution from which will
        be sampled with probability p.
    Returns
    -------
    Object of type ZIDist
        Zero-inflated version of the base distribution.
    '''

    class ZIDist(tfd.Mixture):
      def __init__(self, probs, *args, **kwargs):
          probs_ext = tf.stack([1 - probs, probs], axis = probs.shape.ndims)
          
          super().__init__(
              cat=tfd.Categorical(probs=probs_ext),
              components=[
                  tfd.Deterministic(loc=tf.zeros_like(probs)),
                  dist(*args, **kwargs)        
              ])    
    
    return ZIDist


TransNormal = transform_param(tfd.Normal, scale=tfp.bijectors.Exp())