from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import *
from keras import models
from utility import *

import tensorflow as tf
import os
import gc

# ______________________________环境配置______________________________
device = tf.config.list_physical_devices('GPU')
if device:
    print("GPU is available!")
    tf.config.set_visible_devices(device[0], 'GPU')
    tf.config.experimental.set_memory_growth(device[0], True)
else:
    print("The GPU is not available, Using CPU!")
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

epoch = 40
encode_seq_path = 'BioGRID/human/encode_full_1500.pkl'
train_path = 'BioGRID/human/train.tsv'  # 训练集
valid_path = 'BioGRID/human/valid.tsv'  # 验证集

X_train, Y_train, label_train = read_data(encode_seq_path, train_path, False)
X_valid, Y_valid, label_valid = read_data(encode_seq_path, valid_path, False)

X_train = np.vstack(X_train)
Y_train = np.vstack(Y_train)
X_valid = np.vstack(X_valid)
Y_valid = np.vstack(Y_valid)

label_train = tf.keras.utils.to_categorical(label_train, num_classes=2)
label_valid = tf.keras.utils.to_categorical(label_valid, num_classes=2)


input_A = tf.keras.Input(shape=(17161,), name='input_A')
input_B = tf.keras.Input(shape=(17161,), name='input_B')

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
    trans = Transformer(32, 4, 512)(x)
    x = Add()([x, trans])
    x = LeakyReLU()(x)

    x = Conv1D(64, kernel_size=3, strides=2, activation='relu')(x)
    x = MaxPooling1D(2)(x)
    trans = Transformer(64, 4, 512)(x)
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

MLP = tf.keras.models.Sequential([
    Dense(1024, activation=LeakyReLU()),
    Dropout(0.1),

    Dense(256, activation=LeakyReLU()),
    Dropout(0.1),

    Dense(2, activation='softmax')
])
model_out = MLP(layer_norm)
model = models.Model(inputs=[input_A, input_B], outputs=model_out)

opt = Adam(learning_rate=6e-4)
loss_fn = tf.keras.losses.BinaryCrossentropy()

model.compile(optimizer=opt,
              loss=loss_fn,
              metrics=['accuracy'])

model.summary()
if not os.path.exists('output'):
    os.mkdir('output')

species = 'human'
model_name = 'LM_' + species + '.h5'

model_checkpoint = ModelCheckpoint(filepath='output/' + model_name,
                                   monitor='val_accuracy',
                                   save_best_only=True,
                                   save_weights_only=False,
                                   verbose=1)

callbacks = [model_checkpoint]
print(f"Start training...{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
train_begin = datetime.now()
for i in range(epoch):
    start_time = datetime.now()
    history = model.fit([X_train, Y_train], label_train,
                        batch_size=256,
                        epochs=1,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=([X_valid, Y_valid], label_valid))
    end_time = datetime.now()
    print(f"Epoch {i + 1} spend time: ", end_time - start_time)
    gc.collect()
print(f"End training...{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
train_end = datetime.now()
print("Training time spent: ", train_end - train_begin)
