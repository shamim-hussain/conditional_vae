

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def kl_gaussians(mu_1,sigma_1, mu_2, sigma_2):
    return tf.math.log(sigma_2/sigma_1) \
            + (sigma_1**2 + (mu_1-mu_2)**2)/(2*sigma_2**2) - 0.5

def kl_std_gaussian(mu, sigma):
    return -tf.math.log(sigma) + 0.5*(sigma**2 + mu**2) - 0.5


class SamplingLoss(layers.Layer):
        
    def call(self,inputs):
        x,mean_q,std_q=inputs
        
        kl_div = kl_std_gaussian(mean_q, std_q)        
        compress_loss = tf.reduce_sum(tf.reduce_mean(kl_div,axis=0))
        
        self.add_loss(compress_loss)
        
        return inputs




class ConditionalSamplingLoss(layers.Layer):        
    def call(self,inputs):
        _, mean_q,std_q, mean_p,std_p = inputs
        
        kl_div = kl_gaussians(mean_q, std_q, mean_p, std_p)
        compress_loss = tf.reduce_sum(tf.reduce_mean(kl_div,axis=0))
        
        self.add_loss(compress_loss)
        
        return inputs



class XELoss(layers.Layer):
    def call(self, inputs):
        y_true, y_pred = inputs
        losses_elem=keras.backend.binary_crossentropy(y_true,y_pred)
        
        loss = tf.reduce_sum(
                tf.reduce_mean(losses_elem,
                                axis=0))
        self.add_loss(loss)
        return y_pred
    
class L1Loss(layers.Layer):
    def call(self, inputs):
        y_true, y_pred = inputs
        losses_elem=tf.abs(y_true - y_pred)
        
        XEloss = tf.reduce_sum(
                tf.reduce_mean(losses_elem,
                                axis=0))
        self.add_loss(XEloss)
        return y_pred
        
