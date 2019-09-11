# test_conv_batchnorm_fusing.py
#
# See also:
# https://tehnokv.com/posts/fusing-batchnorm-and-conv/

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Conv2D, SeparableConv2D, BatchNormalization
from tensorflow.keras import Model

with tf.device('/CPU:0'):

        # How can weights of convolution layer and a batchnorm layer
        # (in that order) be combined into a single convolution layer?
        # How can it be done with a separable convolution?

        tf.random.set_seed(1)

        input_shape = (100, 100, 3)
        chan_out = 16
        padding = 'same' # same or valid

        # Build regular model
        def build_full():
            x = Input(shape=input_shape)
            y = Conv2D(chan_out, (3,3), padding=padding, name='bnconv_conv_1')(x)
            y = BatchNormalization(axis=-1, name='bnconv_bn_1', fused=False,
                                   beta_initializer=tf.initializers.RandomUniform(0, 1),
                                   gamma_initializer=tf.initializers.RandomNormal(0, 0.5),
                                   moving_mean_initializer=tf.initializers.RandomNormal(0, 0.5),
                                   moving_variance_initializer=tf.initializers.RandomUniform(0, 1))(y)
            model = Model(x, y)
            return model

        # Build composite model
        def build_composite():
            x = Input(shape=input_shape)
            y = Conv2D(chan_out, (3,3), padding=padding, name='bnconv_comp_1')(x)
            model = Model(x, y)
            return model

        model_full = build_full()
        model_composite = build_composite()

        # Code to integrate batchnorm params into conv layer params
        names = [layer.name for layer in model_full.layers]
        for name in names:
            if name.startswith('bnconv_bn'):
                name_bn = name
                name_conv = 'bnconv_conv' + name[len('bnconv_bn'):]
                name_comp = 'bnconv_comp' + name[len('bnconv_bn'):]

                layer_bn = model_full.get_layer(name=name_bn)
                params_bn = layer_bn.get_weights()
                gamma = params_bn[0]
                beta = params_bn[1]
                moving_mean = params_bn[2]
                moving_variance = params_bn[3]
                epsilon = layer_bn.epsilon

                m_bn = gamma / np.sqrt(moving_variance + epsilon)
                b_bn = beta - m_bn * moving_mean

                ## Compute new convolution kernel of composite layer/model
                layer_conv = model_full.get_layer(name=name_conv)
                params_conv = layer_conv.get_weights()
                w_conv = params_conv[0]
                b_conv = params_conv[1]

                b_comp = m_bn * b_conv + b_bn
                m_bn = np.reshape(m_bn, (1, 1, 1,  m_bn.size))
                w_comp = w_conv * m_bn

                ## Compute new convolution bias of composite layer/model
                # Mimic impact of convolution on the shift/offset term of the batchnorm.
                # This is slightly incorrect at the x/y spatial edges of the tensors.
                # w_conv_sum = np.sum(w_conv, axis=(0,1))    # b_bn has no x/y spatial dependence, so sum conv weights in x and y.
                # b_bnconv = np.sum(w_conv_sum * np.reshape(b_bn, (b_bn.size, 1)), axis=0) # mimic the convolution
                # b_comp = b_conv + b_bnconv

                # Set parameters for composite layer/model
                layer_comp = model_composite.get_layer(name=name_comp)
                layer_comp.set_weights([w_comp, b_comp])


        # Put data through models and measure the output delta
        n_samples = 1
        x = tf.random.normal([n_samples] + list(input_shape))

        y_full = model_full.predict(x)
        y_composite = model_composite.predict(x)

        delta = y_full - y_composite
        mae = np.mean(np.absolute(delta))
        print('MAE: %0.2e' % (mae))

        # Show delta
        delta = delta[0]
        delta = np.sum(delta, axis=2)

        import matplotlib.pyplot as plt
        plt.ion()
        plt.figure(1)
        plt.imshow(delta)
        plt.colorbar()
