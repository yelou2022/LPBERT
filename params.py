import os
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.metrics import *
from tensorflow.keras.losses import BinaryCrossentropy
import numpy as np
from bayes_opt import BayesianOptimization
from utility import Transformer, read_data

device = tf.config.list_physical_devices('GPU')
if device:
    print("GPU is available!")
    tf.config.set_visible_devices(device[7], 'GPU')
    tf.config.experimental.set_memory_growth(device[7], True)
else:
    print("The GPU is not available, Using CPU!")
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

encode_seq_path = 'data/Human/encode_full_1500.pkl'
train_path = 'data/Human/train.txt'
valid_path = 'data/Human/valid.txt'

X_train, Y_train, label_train = read_data(encode_seq_path, train_path, False)
X_valid, Y_valid, label_valid = read_data(encode_seq_path, valid_path, False)

X_train = np.vstack(X_train)
Y_train = np.vstack(Y_train)
X_valid = np.vstack(X_valid)
Y_valid = np.vstack(Y_valid)

label_train = tf.keras.utils.to_categorical(label_train, num_classes=2)
label_valid = tf.keras.utils.to_categorical(label_valid, num_classes=2)

def build_model(learning_rate, optimizer, num_head1, num_head2, ffn_dim1, ffn_dim2, dropout_rate):
    input_A = Input(shape=(17161,), name='input_A')
    input_B = Input(shape=(17161,), name='input_B')

    local_A = tf.reshape(input_A[:, :1562], [-1, 71, 22])
    global_A = tf.reshape(input_A[:, 1562:], [-1, 821, 19])

    local_B = tf.reshape(input_B[:, :1562], [-1, 71, 22])
    global_B = tf.reshape(input_B[:, 1562:], [-1, 821, 19])

    def global_block(input_tensor):
        x = Conv1D(64, kernel_size=7, strides=2, padding='same')(input_tensor)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Conv1D(32, kernel_size=3, strides=2, activation='relu')(x)
        x = MaxPooling1D(pool_size=3, strides=2, padding='same')(x)
        trans = Transformer(32, num_head1, ffn_dim1)(x)
        x = Add()([x, trans])
        x = LeakyReLU()(x)

        x = Conv1D(64, kernel_size=3, strides=2, activation='relu')(x)
        x = MaxPooling1D(2)(x)
        trans = Transformer(64, num_head2, ffn_dim2)(x)
        x = Add()([x, trans])
        x = LeakyReLU()(x)
        x = Flatten()(x)

        return x

    def local_block(input_tensor):
        x = Conv1D(32, kernel_size=3, strides=2, padding='same', activation='relu')(input_tensor)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=3, strides=2, padding='same')(x)
        lstm_x = Bidirectional(LSTM(16, return_sequences=True))(x)
        x = Add()([x, lstm_x])
        x = LeakyReLU()(x)

        x = Conv1D(64, kernel_size=3, strides=2, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=3, strides=2, padding='same')(x)
        lstm_x = Bidirectional(LSTM(32, return_sequences=True))(x)
        x = Add()([x, lstm_x])
        x = LeakyReLU()(x)
        x = Flatten()(x)

        return x

    global_GA = global_block(global_A)
    global_GB = global_block(global_B)
    local_LA = local_block(local_A)
    local_LB = local_block(local_B)

    concatenated = Concatenate(axis=1)([global_GA, global_GB, local_LA, local_LB])

    layer_norm = LayerNormalization()(concatenated)

    MLP = Sequential([
        Dense(1024, activation=LeakyReLU()),
        Dropout(dropout_rate),

        Dense(256, activation=LeakyReLU()),
        Dropout(dropout_rate),

        Dense(2, activation='softmax')
    ])
    model_out = MLP(layer_norm)
    model = Model(inputs=[input_A, input_B], outputs=model_out)

    opt = optimizer(learning_rate=learning_rate)
    loss_fn = BinaryCrossentropy()

    model.compile(optimizer=opt,
                  loss=loss_fn,
                  metrics=['accuracy'])

    return model

pbounds = {'learning_rate': (1e-4, 1e-3),
           'optimizer': (0, 1),  # 0 for 'Adam', 1 for 'SGD'
           'num_head1': (2, 8),
           'num_head2': (2, 8),
           'ffn_dim1': (64, 512),
           'ffn_dim2': (64, 512),
           'dropout_rate': (0.0, 0.3)}

optimizer_list = [tf.keras.optimizers.Adam, tf.keras.optimizers.SGD]

def fitness(learning_rate, optimizer, num_head1, num_head2, ffn_dim1, ffn_dim2, dropout_rate):
    optimizer = int(round(optimizer))
    num_head1 = int(round(num_head1))
    ffn_dim1 = int(round(ffn_dim1))
    num_head2 = int(round(num_head2))
    ffn_dim2 = int(round(ffn_dim2))

    opt = optimizer_list[optimizer]
    model = build_model(learning_rate, opt, num_head1, num_head2, ffn_dim1, ffn_dim2, dropout_rate)
    history = model.fit([X_train, Y_train], label_train,
                        epochs=15,
                        batch_size=256,
                        validation_data=([X_valid, Y_valid], label_valid),
                        verbose=0)
    val_acc = history.history['val_accuracy'][-1]
    print(f"val_acc: {val_acc}")
    return val_acc

optimizer = BayesianOptimization(f=fitness, pbounds=pbounds, random_state=42)

optimizer.maximize(init_points=5, n_iter=10)

print(optimizer.max)
