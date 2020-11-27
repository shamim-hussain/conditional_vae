

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp


# def kl_gaussians(mu_1,sigma_1, mu_2, sigma_2):
#     return tf.math.log(sigma_2/sigma_1) \
#             + (sigma_1**2 + (mu_1-mu_2)**2)/(2*sigma_2**2) - 0.5



class SamplingLoss(layers.Layer):
    def build(self,input_shapes):
        _,mean_shape,std_shape=input_shapes
        self.prior=tfp.distributions.Normal(tf.zeros(mean_shape[-1]),
                                            tf.ones(std_shape[-1]))
        self.built=True
        
    def call(self,inputs):
        x,mean,std=inputs
        posterior=tfp.distributions.Normal(mean, tf.abs(std)+1e-9)
        
        compress_loss = tf.reduce_sum(tf.reduce_mean(
            tfp.distributions.kl_divergence(posterior,self.prior),axis=0))
        
        self.add_loss(compress_loss)
        
        return x,mean,std




class ConditionalSamplingLoss(layers.Layer):        
    def call(self,inputs):
        _, mean_q,std_q, mean_p,std_p = inputs
        prior = tfp.distributions.Normal(mean_p, tf.abs(std_p)+1e-9)
        posterior = tfp.distributions.Normal(mean_q, tf.abs(std_q)+1e-9)
        
        compress_loss = tf.reduce_sum(tf.reduce_mean(
            tfp.distributions.kl_divergence(posterior,prior),axis=0))
        
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
        
