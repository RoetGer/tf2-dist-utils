import inspect
from functools import wraps

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

def transform_param(cls, **transf_dics):
    '''Transform parameters of a class __init__ method
    
    Parameters
    ----------
    cls: class 
        Class whose __init__ method should be wrapped.
    transf_dics: dict
        Dictionary containing parameters to be transformed as keys
        and the respective transformations as value.
    
    Returns
    -------
    Wrapped class with transformed __init__ method.
    '''

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
    Object of type ´ZIDist´ 
        Zero-inflated version of the base distribution.
    '''
    sig = inspect.signature(dist.__init__)

    # The [1:] removes the self from the signature
    f_header_str = ", ".join([
        k if v.default is inspect.Parameter.empty else str(v)
          for k, v in list(sig.parameters.items())[1:]
    ])

    f_call_str = ", ".join(list(sig.parameters.keys())[1:])

    # class ZIDist(tfd.Mixture):
    class_str = f'''
    class ZIDist:
      def __init__(self, dist):
          self.dist = dist
          
      def __call__(self, probs, {f_header_str}, *args, **kwargs):
          probs_ext = tf.stack([1 - probs, probs], axis = probs.shape.ndims)
          
          mixt = tfd.Mixture(
              cat=tfd.Categorical(probs=probs_ext),
              components=[
                  tfd.Deterministic(loc=tf.zeros_like(probs)),
                  self.dist({f_call_str}, *args, **kwargs)        
              ])
          
          return mixt
    '''

    # Create class - cleandoc removes the excess indentation
    exec(inspect.cleandoc(class_str))

    return eval("ZIDist(dist)")

class ZIBuilder:
    def __init__(self, dist):
        self.dist = dist

    def __call__(self, *args, **kwargs):
      dist = self.dist

      class ZIDist(tfd.Mixture):
        def __init__(self, probs, *args, **kwargs):
            probs_ext = tf.stack([1 - probs, probs], axis = probs.shape.ndims)
            
            super().__init__(
                cat=tfd.Categorical(probs=probs_ext),
                components=[
                    tfd.Deterministic(loc=tf.zeros_like(probs)),
                    dist(*args, **kwargs)        
                ])
      return ZIDist(*args, **kwargs)


# Some example zero-inflated distributions
ZINormal = build_zero_infl_dist("ZINormal", tfd.Normal)
ZIPoisson = build_zero_infl_dist("ZIPoisson", tfd.Poisson) 


# Some example transformed distributions
TransNormal = transform_param(tfd.Normal, scale=tfp.bijectors.Exp())
TransZINormal = transform_param(
    ZINormal,
    probs=tfp.bijectors.SoftClip(low=0., high=1.),
    scale=tfp.bijectors.Exp())
TransZIPoisson = transform_param(
    ZIPoisson, 
    probs=tfp.bijectors.SoftClip(low=0., high=1.),
    rate=tfp.bijectors.Exp())