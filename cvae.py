
import tensorflow as tf
from tensorflow.keras import layers,optimizers,models,callbacks
import numpy as np

from vae_layers import (ConditionalSamplingLoss, XELoss,
                        Sampling, Reparameterize)





class Encoder(layers.Layer):
    def __init__(self, num_classes, emb_dim, hid_dim, lat_dim, **kwargs):
        super().__init__(**kwargs)
        
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.lat_dim = lat_dim
        
        self.emb_x = layers.Dense(self.emb_dim)
        self.emb_c = layers.Embedding(self.num_classes,self.emb_dim)
        
        self.dense_h = layers.Dense(self.hid_dim)
        self.dense_l = layers.Dense(self.lat_dim*2)
    
    def call(self,inputs):
        in_x, in_c = inputs
        
        in_x = tf.reshape(in_x, [-1, np.prod(in_x.shape.as_list()[1:])])
        
        emb = tf.nn.relu(self.emb_x(in_x) + self.emb_c(in_c))
        hid = tf.nn.relu(self.dense_h(emb))
        
        lat = self.dense_l(hid)
        mu_q = lat[..., :self.lat_dim]
        sigma_q = tf.exp(lat[..., self.lat_dim: ])
        
        return mu_q, sigma_q



class Prior(layers.Layer):
    def __init__(self, num_classes, lat_dim, **kwargs):
        super().__init__(**kwargs)
        
        self.num_classes = num_classes
        self.lat_dim = lat_dim
        
        self.emb_c = layers.Embedding(self.num_classes, self.lat_dim*2)
    
    def call(self, inputs):
        in_c = inputs
        
        emb = self.emb_c(in_c)
        mu_p = emb[..., :self.lat_dim]
        sigma_p = tf.exp(emb[..., self.lat_dim: ])
        
        return mu_p, sigma_p




class Decoder(layers.Layer):
    def __init__(self, num_classes, emb_dim, hid_dim, out_dim, **kwargs):
        super().__init__(**kwargs)
        
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.out_dim = tuple(out_dim)
    
        self.emb_z = layers.Dense(self.emb_dim)
        self.emb_c = layers.Embedding(self.num_classes,self.emb_dim)
        
        self.dense_h = layers.Dense(self.hid_dim)
        self.dense_o = layers.Dense(np.prod(self.out_dim))
    
    def call(self, inputs):
        in_z, in_c = inputs
        
        emb = tf.nn.relu(self.emb_z(in_z) + self.emb_c(in_c))
        hid = tf.nn.relu(self.dense_h(emb))
        
        out = tf.nn.sigmoid(self.dense_o(hid))
        out = tf.reshape(out, (-1,)+self.out_dim)
        
        return out



class PlotSamples(callbacks.Callback):
    def __init__(self, decoder_model, lat_dim, num_classes=10,
                 n_row=10, n_col=10,
                 freq=10, figsize=(10,10)):
        self.decoder_model = decoder_model
        self.lat_dim = lat_dim
        self.num_classes = num_classes
        self.n_row = n_row
        self.n_col = n_col
        self.freq = freq
        
        self.num_samples = self.n_row*self.n_col
    
    def on_epoch_end(self, epoch, logs=None):
        import matplotlib.pyplot as plt
        if epoch % self.freq:
            return
        
        e = np.random.randn(self.num_samples,self.lat_dim)
        c = (np.arange(self.num_samples) % self.num_classes).astype(np.float32)
        
        out = self.decoder_model.predict_on_batch([e,c])
        
        img = out.reshape(self.n_row, self.n_col, *out.shape[1:])\
                .transpose(0,2,1,3).reshape(self.n_row*out.shape[1],
                                            self.n_col*out.shape[2])
        
        plt.figure(figsize=(10,10))
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.title(f'Epoch:{epoch}')
        
        plt.show()
        plt.pause(0.1)
        
        figs=plt.get_fignums()
        if len(figs)>5: 
            for ff in figs[0:len(figs)-5]: plt.close(ff)
        


class CVAE:
    def __init__(self, in_out_shape, num_classes, lat_dim, 
                 enc_emb_dim, enc_hid_dim, 
                 dec_emb_dim, dec_hid_dim):
        
        self.in_out_shape = in_out_shape
        self.num_classes = num_classes
        self.lat_dim = lat_dim
        self.enc_emb_dim = enc_emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_emb_dim = dec_emb_dim
        self.dec_hid_dim = dec_hid_dim
        
        self.encoder_layer = Encoder(self.num_classes, 
                                     self.enc_emb_dim, self.enc_hid_dim, 
                                     self.lat_dim)
        self.decoder_layer = Decoder(self.num_classes, 
                                     self.dec_emb_dim, self.dec_hid_dim,
                                     self.in_out_shape)
        self.prior_layer = Prior(self.num_classes, self.lat_dim)
    
    def get_encoder_model(self):
        if hasattr(self, 'encoder_model'):
            return self.encoder_model
        
        in_x = layers.Input(self.in_out_shape)
        in_c = layers.Input( () )
        
        mu_q, sigma_q = self.encoder_layer([in_x, in_c])
        self.encoder_model = models.Model([in_x, in_c], [mu_q, sigma_q])
        
        return self.encoder_model
    
    def get_decoder_model(self):
        if hasattr(self, 'decoder_model'):
            return self.decoder_model
        
        
        in_e = layers.Input((self.lat_dim,))
        in_c = layers.Input( () )
        
        mu_p, sigma_p = self.prior_layer(in_c)
        z = Reparameterize()([in_e,mu_p,sigma_p])
        
        out = self.decoder_layer([z, in_c])
        
        self.decoder_model = models.Model([in_e,in_c],out)
        return self.decoder_model
    
    def get_autoencoder_model(self):
        if hasattr(self, 'autoencoder_model'):
            return self.autoencoder_model
        
        in_x = layers.Input(self.in_out_shape)
        in_c = layers.Input( () )
        
        mu_q, sigma_q = self.encoder_layer([in_x, in_c])
        mu_p, sigma_p = self.prior_layer(in_c)
        
        z = Sampling()([mu_q,sigma_q])
        z,*_ = ConditionalSamplingLoss()([z,mu_q,sigma_q,mu_p,sigma_p])
        
        out = self.decoder_layer([z, in_c])
        out = XELoss()([in_x, out])
        
        self.autoencoder_model = models.Model([in_x,in_c],out)
        
        return self.autoencoder_model
                



def load_data():
    from tensorflow.keras.datasets import mnist
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train = X_train.astype('float32')/255.
    X_test = X_test.astype('float32')/255.
    return (X_train, Y_train), (X_test, Y_test)


def main(   lat_dim         = 4,
            batch_size      = 256,
            learning_rate   = 2.5e-4,
            num_epochs      = 1000,
            dim_big         = 512,
            dim_small       = 128  ):
    
    (X_train, Y_train), (X_test, Y_test) = load_data()
    
    vae = CVAE(in_out_shape = X_train.shape[1:],
              num_classes = Y_train.max()+1, 
              lat_dim = lat_dim,
              enc_emb_dim = dim_big, enc_hid_dim = dim_small,
              dec_emb_dim = dim_small, dec_hid_dim = dim_big)
    
    autoencoder_model = vae.get_autoencoder_model()
    autoencoder_model.summary()
    
    opt = optimizers.Adam(learning_rate)
    autoencoder_model.compile(opt, None)
    
    cbacks = []
    cbacks.append(PlotSamples(vae.get_decoder_model(), lat_dim))
    
    autoencoder_model.fit([X_train,Y_train], None,
                          batch_size=batch_size, epochs=num_epochs, 
                          validation_data=([X_test,Y_test], None), 
                          callbacks=cbacks)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-lat_dim', type=int, default=4)
    parser.add_argument('-batch_size', type=int, default=256)
    parser.add_argument('-learning_rate', type=float, default=2.5e-4)
    parser.add_argument('-num_epochs', type=int, default=1000)
    parser.add_argument('-dim_big', type=int, default=512)
    parser.add_argument('-dim_small', type=int, default=128)
    
    config = parser.parse_args()
    main(**config.__dict__)





