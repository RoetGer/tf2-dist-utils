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
        
        lst = tf.split(
            y_pred, 
            num_or_size_splits=shape[-1], 
            axis=(shape.ndims - 1))
        rv = self.dist(*lst)

        return -rv.log_prob(y_true)

def build_loss(class_loss_name, dist):
    '''Creates a tensorflow 2.x loss function based on NegLokLikeLoss

    Parameters
    ----------
    class_loss_name : str
        Name/Type of the loss to be created, e.g. "TransNormal"
    dist : distribution object
        Can be any type of tensorflow distribution object, 
        which supports parameter specification via call 
        (e.g. dist(param1, param2)) and has a log_prob method
    
    Returns
    -------
    Object of type class_loss_name
        Loss object of type class_loss_name which can be used
        in conjuction with tensorflow 2.x models.
    '''
    
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


NegGaussLogLikeLoss = build_loss("NegGaussLogLikeLoss", TransNormal)