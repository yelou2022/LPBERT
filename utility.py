import pickle
import numpy as np

from datetime import datetime
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers


class Transformer(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(Transformer, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super(Transformer, self).get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
        })
        return config


def truncation_sequence(seq_list, seq_len):
    """
    :param seq_list:
    :param seq_len:
    :return:
    """
    for i in range(len(seq_list)):
        if len(seq_list[i]) > seq_len:
            seq_list[i] = seq_list[i][:seq_len]
    return seq_list


def read_dataset_for_generator(seq_dict_path, dataset_path, seq_len, shuffle):
    """
    :param seq_dict_path:
    :param dataset_path:
    :return: protein1,protein2,label
    """
    protein1_list = []
    protein2_list = []
    label_list = []
    seq_dict = pickle.load(open(seq_dict_path, 'rb'))
    datas = pickle.load(open(dataset_path, 'rb'))
    if shuffle:
        np.random.seed(200)
        for i in range(10):
            np.random.shuffle(datas)

    for data in datas:
        protein1, protein2, label = data[0].split()
        try:
            protein1_list.append(seq_dict[protein1][:seq_len])
            protein2_list.append(seq_dict[protein2][:seq_len])
            label_list.append(int(label))
        except KeyError:
            return "error: can not correctly get the protein sequence from the dictionary!"

    return protein1_list, protein2_list, label_list


def read_data(encode_seq_dict_path, dataset_path, shuffle):
    """
    :param encode_seq_dict_path:
    :param dataset_path:
    :return: protein1,protein2,label
    """
    protein1_list = []
    protein2_list = []
    label_list = []
    seq_dict = pickle.load(open(encode_seq_dict_path, 'rb'))
    with open(dataset_path, 'r') as f:
        datas = f.readlines()
    if shuffle:
        np.random.seed(200)
        for i in range(10):
            np.random.shuffle(datas)
    for data in datas:
        protein1, protein2, label = data.rstrip('\n').split('\t')
        try:
            protein1_list.append(seq_dict[protein1])
            protein2_list.append(seq_dict[protein2])
            label_list.append(int(label))
        except KeyError:
            return "error: can not correctly get the protein sequence from the dictionary!"

    return protein1_list, protein2_list, label_list


def get_metric(true_label, predict_label):
    """
    :param true_label:
    :param predict_label:
    :return:
    """
    cm = confusion_matrix(true_label, predict_label)
    tp = cm[1][1]
    fp = cm[0][1]
    tn = cm[0][0]
    fn = cm[1][0]
    print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
    return tp, fp, tn, fn


def evaluate_model(tp, fp, tn, fn):
    """
    :param tp:
    :param fp:
    :param tn:
    :param fn:
    :return:
    """
    accuracy = (tp + tn) / (tp + fp + tn + fn) if tp + fp + tn + fn != 0 else np.nan
    precision = tp / (tp + fp) if tp + fp != 0 else np.nan
    recall = tp / (tp + fn) if tp + fn != 0 else np.nan
    specificity = tn / (tn + fp) if fp + tn != 0 else np.nan
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall != 0 else np.nan
    mcc = (tp * tn - fp * fn) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5 if (tp + fp) * (tp + fn) * (
            tn + fp) * (tn + fn) != 0 else np.nan
    return accuracy, precision, recall, specificity, f1_score, mcc


def compute(label, label_hat):
    label_hat = np.argmax(label_hat, axis=1)
    tp, fp, tn, fn = get_metric(label, label_hat)
    acc, pre, recall, spec, f1_score, mcc = evaluate_model(tp, fp, tn, fn)
    print(f"result：acc:{acc}, precision:{pre}, recall:{recall}, specificity:{spec},"
          f" f1_score:{f1_score}, mcc:{mcc}")


def seq_encode(seq_path, seq_len, protein_bert, input_encoder):
    seq_data = pickle.load(open(seq_path, 'rb'))
    encoded_seq_data = {}
    print(f"Start encoding...{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    for key, value in seq_data.items():
        seq = [value[:seq_len]]
        encoded_x = input_encoder.encode_X(seq, seq_len + 2)
        local_representations, _ = protein_bert.predict(encoded_x, batch_size=1)
        encoded_seq_data[key] = local_representations[0]

    print(f"End encoding...{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return encoded_seq_data


def read_dict_sequence(seq_dict_path):
    """
    :param seq_dict_path:
    :return:
    """
    with open(seq_dict_path, 'r') as f:
        seq_dict = {}
        for line in f:
            line = line.strip().split('\t')
            seq_dict[line[0]] = line[1]
    return seq_dict


if __name__ == '__main__':
    pass
