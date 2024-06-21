from proteinbert import load_pretrained_model
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs
from utility import *
import tensorflow as tf
import os

device = tf.config.list_physical_devices('GPU')
if device:
    print("The GPU is available!")
    tf.config.set_visible_devices(device[0], 'GPU')
    tf.config.experimental.set_memory_growth(device[0], True)
else:
    print("The GPU is not available, Using CPU!")
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

seq_data = pickle.load(open('./physical/human/seqs.pkl', 'rb'))
# seq_len=500, 1000, 1500
seq_len = 1500

# seq_data = read_dict_sequence('./data/pan/pan_dict.tsv')
# seq_len = 1500

pretrained_model_generator, input_encoder = load_pretrained_model()
model = get_model_with_hidden_layers_as_outputs(pretrained_model_generator.create_model(seq_len + 2))
encoded_seq_data = {}
i = 1
start = datetime.now()
print(f"encode start...", start.strftime('%Y-%m-%d %H:%M:%S'))
for key, value in seq_data.items():
    seq = [value[:seq_len]]
    encoded_x = input_encoder.encode_X(seq, seq_len + 2)
    local_res, global_res = model.predict(encoded_x, batch_size=1)
    print(f"{i}: {key} encode finished, shape: {local_res.shape}, {global_res.shape}")

    local_ = local_res[0]
    global_ = global_res[0]

    local_ = tf.reduce_max(local_, axis=0)

    res = tf.concat([local_, global_], axis=0)

    encoded_seq_data[key] = res
    i += 1

end = datetime.now()
print(f"encode end...", end.strftime('%Y-%m-%d %H:%M:%S'))
print(f"encode cost time: {end - start}")
pickle.dump(encoded_seq_data, open(f'./physical/human/encode_full_{seq_len}.pkl', 'wb'))
