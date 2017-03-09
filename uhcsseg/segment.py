import numpy as np
from sklearn.linear_model import SGDClassifier

class TensorSGD():
    """ fit SGD-SVM on hc batch data -- takes a couple seconds for 4 images
    SVM with LibLinear takes quite a while for the same dataset
    """
    def __init__(self, loss='hinge', alpha=0.0001, class_weight='balanced'):
        self.loss = loss
        self.alpha = alpha
        self.class_weight=class_weight
        self.clf = SGDClassifier(
            loss=self.loss,
            alpha=self.alpha,
            class_weight=self.class_weight
        )

    def fit(self, Xtrain, y):
        # reshape feature map into [feature, channels]
        # expect Xtrain to be a keras tensor
        # y should be a list of label images
        ntrain, nchan, h, w = Xtrain.shape

        X = Xtrain.transpose(0,2,3,1) # to [batch, height, width, channels]
        X = X.reshape((-1, nchan)) # to [feature, channels]
        print(X.shape)
        y = np.array(y).flatten()

        # instance normalization: 
        X = X / np.linalg.norm(X, axis=1)[:,np.newaxis]
        self.clf.fit(X, y)
        print('SVM model finished.')

    def predict(self, X):
        n, nchan, h, w = X.shape
        X = X.transpose(0,2,3,1) # to [batch, height, width, channels]
        X = X.reshape((-1, nchan)) # to [feature, channels]
        
        # instance normalization: 
        X = X / np.linalg.norm(X, axis=1)[:,np.newaxis]
        
        y_pred = self.clf.predict(X)
        yy = y_pred.reshape((n, h, w))
        return yy
