import keras.backend as K

def focal_crossentropy_loss(focus_param=2, balance_param=0.25, average=False, class_weights=None):
    """ arXiv:1708.02002 
    focus_param: gamma
    balance_param: alpha
    
    consider setting alpha to inverse class frequency
    """

    if class_weights is not None:
        class_weights = K.variable(class_weights)
    
    def loss(target, output):
        log_pt = -K.categorical_crossentropy(target, output)
        pt = K.exp(log_pt)
        focal_loss = -((1 - pt) ** focus_param) * log_pt
        
        if class_weights is None:
            balanced_focal_loss = balance_param * focal_loss
        else:
            y_true = K.argmax(target, axis=-1)
            alpha_t = K.gather(class_weights, y_true)
            balanced_focal_loss = alpha_t * focal_loss
            
        if average:
            return K.mean(balanced_focal_loss)
        return K.sum(balanced_focal_loss)

    return loss
