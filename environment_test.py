from proteinbert import load_pretrained_model
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs
import tensorflow as tf
import os

device = tf.config.list_physical_devices('GPU')
print(tf.__version__)
print(device)

if device:
    print("GPU is available!")
    tf.config.set_visible_devices(device[0], 'GPU')
    tf.config.experimental.set_memory_growth(device[0], True)
else:
    print("The GPU is not available, Using CPU!")
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

seqs = ['MKTVRQERLKSIVRILERSKEPVSGAQLA', 'EELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG']

seq_len = 36
batch_size = 1

pretrained_model_generator, input_encoder = load_pretrained_model()
model = get_model_with_hidden_layers_as_outputs(pretrained_model_generator.create_model(seq_len + 2))
encoded_x = input_encoder.encode_X(seqs, seq_len + 2)
local_, global_ = model.predict(encoded_x, batch_size=batch_size)

print(f"encoded_x:", encoded_x)

print(local_.shape)
print(global_.shape)
