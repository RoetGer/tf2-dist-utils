import tensorflow as tf

class NegLogLikeLoss(tf.keras.losses.Loss):

    def call(self, y_true, y_pred):
        shape = y_pred.shape
        
        lst = tf.split(
            y_pred, 
            num_or_size_splits=shape[-1], 
            axis=(shape.ndims - 1))
        rv = self.dist(*lst)

        return -rv.log_prob(y_true)

def build_loss(class_loss_name, dist):
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