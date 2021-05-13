import math, random
from const import START, STOP

import numpy as np
from collections import defaultdict, OrderedDict
from pprint import pprint

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import Layer
from paddle.nn import MultiHeadAttention, Sequential, Linear, LayerNorm, Dropout, ReLU, Conv1D, LSTM
import utils
from paddle.nn.functional import layer_norm




class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.fc1 = Linear(embed_dim, ff_dim)
        self.act1 = ReLU()
        self.fc2 = Linear(ff_dim, embed_dim)
        # self.ffn = Sequential(
        #     (Linear(embed_dim, ff_dim, name='fc1'), ReLU(name='act1'), Linear(ff_dim, embed_dim, name='fc2'))
        # )
        self.layernorm1 = LayerNorm(normalized_shape=(embed_dim), epsilon=1e-6)
        self.layernorm2 = LayerNorm(normalized_shape=(embed_dim), epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def forward(self, inputs):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.fc1(out1)
        ffn_output = self.act1(ffn_output)
        ffn_output = self.fc2(ffn_output)
        # ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)


class Network(Layer):
    def __init__(self, N_neuron=128, N_neuron2=128, N_trans=1, N_lstm=2, N_head=2):
        super(Network, self).__init__()
        self.N_neuron = N_neuron
        self.N_neuron2 = N_neuron2
        self.N_trans = N_trans
        self.N_lstm = N_lstm
        self.N_head = N_head
        self.conv1 = Conv1D(in_channels=10, out_channels=N_neuron, kernel_size=9, padding='SAME', data_format='NLC')
        self.conv2 = Conv1D(in_channels=N_neuron2 * 2 + 1, out_channels=1, kernel_size=1, padding='SAME', data_format='NLC')
        self.lstm = LSTM(input_size=N_neuron, hidden_size=N_neuron2, num_layers=N_lstm, direction='bidirect', dropout=0.25)
        self.lstm2 = LSTM(input_size=N_neuron2 * 2, hidden_size=N_neuron2, direction='bidirect', dropout=0.25)


    def forward(self, x1, x2, x3):
        print(x1.shape, x2.shape, x3.shape)
        x1 = fluid.layers.unsqueeze(input=x1, axes=[2])
        # x1 = fluid.layers.reshape(x1, [-1, 500,  1])
        print('x1reshape:', x1.shape)
        x1 = fluid.layers.one_hot(x1, 5)
        print('x1onehot:', x1.shape)
        # x2 = fluid.layers.reshape(x2, [-1,500, 1])
        x2 = fluid.layers.unsqueeze(input=x2, axes=[2])
        x2 = fluid.layers.one_hot(x2, 4)
        # x3 = fluid.layers.reshape(x3, [-1,500,  1])
        x3 = fluid.layers.unsqueeze(input=x3, axes=[2])
        x = fluid.layers.concat(input=[x1, x2, x3], axis=-1)
        # x = fluid.layers.reshape(x, [-1,500,  10])
        print('concat:', x.shape)
        x = self.conv1(x)
        x = fluid.layers.sigmoid(x)
        # x = fluid.layers.sequence_conv(x,  num_filters=self.N_neuron, filter_size=9, padding_start=-1, act='sigmoid')
        print('conv1:', x.shape)

        # x = layer_norm(x, normalized_shape=x.shape[1:])
        x = fluid.layers.swish(x)
        x = fluid.layers.dropout(x, 0.15)
        for i in range(self.N_trans):
            transformer_block = TransformerBlock(self.N_neuron, self.N_head, 256)
            x = transformer_block(x)
        print('trans:', x.shape)
        print(type(x))
        # for i in range(self.N_lstm):
        x, _ = self.lstm(x)
            # x = paddle.fluid.layers.concat(input=[fwd, back], axis=-1)
        print('lstm:', x.shape)
            # emb = paddle.fluid.layers.fc(emb, size=self.model_size * 2)
        x, _ = self.lstm2(x)
        print('lstm2:', x.shape)

        x = fluid.layers.concat([x, x3], axis=-1)
        print('concat2:', x.shape)
        # x = fluid.layers.sequence_conv(x,  num_filters=1, filter_size=1, padding_start=-1, act='sigmoid')
        x = self.conv2(x)
        x = fluid.layers.sigmoid(x)
        print('conv2:', x.shape)
        # x = fluid.layers.sigmoid(x)
        x = fluid.layers.flatten(x)
        print('out:', x.shape)

        return x




'''
def make_model(N_neuron=128, N_neuron2=128, N_trans=1, N_lstm=2, N_head=2):
    Input1 = fluid.(shape=(None,))
    inp = tf.keras.layers.Masking(mask_value=0.)(Input1)
    x = tf.one_hot(tf.cast(inp, tf.int32), 5)

    Input2 = keras.Input(shape=(None,))
    inp2 = tf.keras.layers.Masking(mask_value=0.)(Input2)
    x2 = tf.one_hot(tf.cast(inp2, tf.int32), 4)

    # Input3 = keras.Input(shape=(None, ))
    # inp3 = tf.keras.layers.Masking(mask_value=0.)(Input3)
    # x3 = tf.one_hot(tf.cast(inp3, tf.int32), 9)

    Input4 = keras.Input(shape=(500, 7))
    inp4 = tf.keras.layers.Masking(mask_value=-2.)(Input4)
    x4 = inp4
    # x4 = layers.Reshape((-1,7))(inp4)

    # Input = keras.Input(shape=(500,32 ))
    # x = Input
    x = layers.concatenate([x, x2, x4], axis=-1)
    print('input:', x.shape)
    x = layers.Conv1D(N_neuron, kernel_size=9, padding='same', activation='sigmoid')(x)
    print('conv1:', x.shape)
    x = layers.LayerNormalization()(x)
    x = layers.Activation('swish')(x)

    x = layers.SpatialDropout1D(0.15)(x)

    for i in range(N_trans):
        transformer_block = TransformerBlock(N_neuron, N_head, 256)
        x = transformer_block(x)
    print('trans:', x.shape)

    for i in range(N_lstm):
        x = layers.Bidirectional(
            layers.LSTM(N_neuron2, dropout=0.25, return_sequences=True, recurrent_initializer="orthogonal",
                        kernel_initializer="orthogonal",
                        ))(x)
    print('lstm:', x.shape)

    x = layers.Bidirectional(
        layers.GRU(N_neuron2, dropout=0.25, return_sequences=True, recurrent_initializer="orthogonal",
                   kernel_initializer="orthogonal",
                   ))(x)

    x = layers.concatenate([x, x4], axis=-1)
    x = layers.Conv1D(1, kernel_size=1, padding='same', activation='sigmoid')(x)
    print('conv2:', x.shape)

    output = layers.Flatten()(x)
    print('output:', output.shape)
    model = keras.Model(inputs=[Input1, Input2, Input4], outputs=output)

    def rmsd(y_true, y_pred):
        mask = tf.cast((y_true != -2), dtype=tf.float32)
        mse = tf.square(y_true - y_pred)

        mse = tf.reduce_sum(mse * mask, axis=1) / tf.reduce_sum(mask, axis=1)
        mse = tf.math.sqrt(mse)
        mse = tf.reduce_mean(mse)

        return mse

    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=rmsd)

    return model

class Network(Layer):
    def __init__(self,
                 sequence_vocabulary, bracket_vocabulary,
                 dmodel=4,
                 n_layers=4,
                 dropout=0.15,
                 n_heads=8,
                 d_ff=2048

                 ):
        super(Network, self).__init__()
        self.sequence_vocabulary = sequence_vocabulary
        self.bracket_vocabulary = bracket_vocabulary
        self.dropout_rate = dropout
        self.model_size = dmodel
        self.n_layers = n_layers

    def forward(self, seq, dot, pos):
        emb_seq = paddle.fluid.embedding(seq, size=(self.sequence_vocabulary.size, self.model_size), is_sparse=True)
        emb_dot = paddle.fluid.embedding(dot, size=(self.bracket_vocabulary.size, self.model_size), is_sparse=True)
        emb = paddle.fluid.layers.concat(input=[emb_seq, emb_dot], axis=1)
        # input_ = paddle.fluid.layers.concat(input=[seq, dot], axis=0)


        # emb_seq = fluid.layers.sequence_conv(seq.reshape((-1, 1)), num_filters=10)
        # emb_dot = fluid.layers.sequence_conv(dot.reshape((-1, 1)), num_filters=10)
        #
        # emb_seq = fluid.layers.sequence_conv(emb_seq,filter_size=5, num_filters=32)
        # emb_dot = fluid.layers.sequence_conv(emb_dot,filter_size=5, num_filters=32)
        #
        # emb = paddle.fluid.layers.concat(input=[emb_seq, emb_dot], axis=1)
        # emb = paddle.fluid.layers.fc(emb, size=self.model_size * 2)
        # # emb = paddle.fluid.layers.dropout(emb, dropout_prob=self.dropout_rate)
        # emb = fluid.layers.batch_norm(emb)
        # emb = paddle.fluid.layers.relu(emb)

        for _ in range(self.n_layers):
            emb = paddle.fluid.layers.fc(emb, size=self.model_size * 8)
            print('shape of lstm_emb1{}: {}'.format(_, emb.shape))
            fwd, cell = paddle.fluid.layers.dynamic_lstm(input=emb, size=self.model_size * 8, use_peepholes=True,
                                                         is_reverse=False)
            back, cell = paddle.fluid.layers.dynamic_lstm(input=emb, size=self.model_size * 8, use_peepholes=True,
                                                          is_reverse=True)
            emb = paddle.fluid.layers.concat(input=[fwd, back], axis=1)

            emb = paddle.fluid.layers.fc(emb, size=self.model_size * 2)
            emb = fluid.layers.batch_norm(emb)
            emb = paddle.fluid.layers.relu(emb)

        ff_out = paddle.fluid.layers.fc(emb, size=4)
        ff_out = fluid.layers.batch_norm(ff_out)
        ff_out = paddle.fluid.layers.relu(ff_out)

        ff_out = paddle.fluid.layers.fc(ff_out, size=2)
        ff_out = fluid.layers.batch_norm(ff_out)
        ff_out = paddle.fluid.layers.relu(ff_out)

        soft_out = paddle.fluid.layers.softmax(ff_out, axis=1)
        return soft_out[:, 0]

    def forward(self, seq, dot, pos):
        emb_seq = paddle.fluid.embedding(seq, size=(self.sequence_vocabulary.size, self.model_size), is_sparse=True)
        emb_seq = paddle.fluid.layers.elementwise_add(emb_seq, pos)
        emb_seq = fluid.layers.concat(input=[emb_seq, emb_seq, emb_seq])
'''