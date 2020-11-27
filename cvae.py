
import tensorflow as tf
from tensorflow.keras import layers,optimizers,models,callbacks
from tensorflow.keras.datasets import mnist
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
        



# Load Data
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.astype('float32')/255.
X_test = X_test.astype('float32')/255.


lat_dim = 4
batch_size = 256
num_epochs = 1000
num_classes = 10
dim_big = 512
dim_small = 128



encoder_layer = Encoder(num_classes, dim_big, dim_small, lat_dim)
decoder_layer = Decoder(num_classes, dim_small, dim_big, X_train.shape[1:])
prior_layer = Prior(num_classes, lat_dim)

# Encoder Model
in_x = layers.Input(X_train.shape[1:])
in_c = layers.Input( () )

mu_q, sigma_q = encoder_layer([in_x, in_c])
encoder_model = models.Model([in_x, in_c], [mu_q, sigma_q])


# Decoder model
in_e = layers.Input((lat_dim,))
in_c = layers.Input( () )

mu_p, sigma_p = prior_layer(in_c)
z = Reparameterize()([in_e,mu_p,sigma_p])

out = decoder_layer([z, in_c])

decoder_model = models.Model([in_e,in_c],out)




# Autoencoder Model
in_x = layers.Input(X_train.shape[1:])
in_c = layers.Input( () )

mu_q, sigma_q = encoder_layer([in_x, in_c])
mu_p, sigma_p = prior_layer(in_c)

z = Sampling()([mu_q,sigma_q])
z,*_ = ConditionalSamplingLoss()([z,mu_q,sigma_q,mu_p,sigma_p])

out = decoder_layer([z, in_c])
out = XELoss()([in_x, out])

autoencoder_model = models.Model([in_x,in_c],out)


autoencoder_model.summary()

opt = optimizers.Adam(5e-4)
autoencoder_model.compile(opt, None)

cbacks = []
cbacks.append(PlotSamples(decoder_model, lat_dim))

autoencoder_model.fit([X_train,Y_train], None,
                      batch_size=batch_size, epochs=num_epochs, 
                      validation_data=([X_test,Y_test], None), 
                      callbacks=cbacks)


# codes=enc_model.predict(X_test,batch_size=512,verbose=2)
# from sklearn.manifold import TSNE
# tsne=TSNE(verbose=2,perplexity=100)
# c2d=tsne.fit_transform(codes)
# plt.figure()
# for k in range(10):
#     x,y=c2d[Y_test==k].T
#     plt.scatter(x,y,marker='x')
# plt.legend(list(map(str,range(10))))