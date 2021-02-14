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


TransNormal = transform_param(tfd.Normal, scale=tfp.bijectors.Exp())