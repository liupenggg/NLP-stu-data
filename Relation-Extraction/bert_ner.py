# %%

import pandas as pd
import numpy as np
#from keras_contrib.layers import CRF
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import tensorflow as tf
#from keras.layers import Dense, Bidirectional, Dropout, LSTM, TimeDistributed, Masking
from bert4keras.layers import ConditionalRandomField
from functools import partial
import scipy as sp
from sklearn.metrics import f1_score
import tensorflow.keras.backend as K
from keras.utils import to_categorical

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
from transformers import *
from load_data import read_data
from utils import train_file_path, dev_file_path

import os, json
print(tf.__version__)

# %%

# https://huggingface.co/bert-base-chinese 去这里下载预训练参数
# %%
with open("%s_label2id.json" % "ccks2019", "r", encoding="utf-8") as h:
    label_id_dict = json.loads(h.read())

id_label_dict = {v:k for k,v in label_id_dict.items()}

BERT_PATH = '../my-bert-base-chinese/'
MAX_SEQUENCE_LENGTH = 128
crf_lr_multiplier = 100  # 必要时扩大CRF层的学习率


# %%

def _convert_to_transformer_inputs(instance, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for transformer (including bert)"""

    def return_id(str1, truncation_strategy, length):
        inputs = tokenizer.encode_plus(str1,
                                       add_special_tokens=True,
                                       max_length=length,
                                       truncation_strategy=truncation_strategy)

        input_ids = inputs["input_ids"]
        input_masks = [1] * len(input_ids)
        input_segments = inputs["token_type_ids"]
        padding_length = length - len(input_ids)
        padding_id = tokenizer.pad_token_id
        input_ids = input_ids + ([padding_id] * padding_length)
        input_masks = input_masks + ([0] * padding_length)
        input_segments = input_segments + ([0] * padding_length)

        return [input_ids, input_masks, input_segments]

    input_ids, input_masks, input_segments = return_id(
        instance, 'longest_first', max_sequence_length)

    return [input_ids, input_masks, input_segments]


def compute_input_arrays(df, tokenizer, max_sequence_length):
    input_ids, input_masks, input_segments = [], [], []
    for instance in tqdm(df):
        ids, masks, segments = \
            _convert_to_transformer_inputs(str(instance), tokenizer, max_sequence_length)

        input_ids.append(ids)
        input_masks.append(masks)
        input_segments.append(segments)

    return [np.asarray(input_ids, dtype=np.int32),
            np.asarray(input_masks, dtype=np.int32),
            np.asarray(input_segments, dtype=np.int32)
            ]


# %%
df_train, train_tags = read_data(train_file_path)
df_test, test_tags = read_data(dev_file_path)

tokenizer = BertTokenizer.from_pretrained(BERT_PATH + 'vocab.txt')
inputs = compute_input_arrays(df_train, tokenizer, MAX_SEQUENCE_LENGTH)
test_inputs = compute_input_arrays(df_test, tokenizer, MAX_SEQUENCE_LENGTH)


# %%

def compute_output_arrays(tags):
    # 对y值统一长度为MAX_SEQ_LEN
    new_y = []
    for seq in tags:
        num_tag = [label_id_dict[_] for _ in seq]
        if len(seq) < MAX_SEQUENCE_LENGTH:
            num_tag = num_tag + [0] * (MAX_SEQUENCE_LENGTH - len(seq))
        else:
            num_tag = num_tag[: MAX_SEQUENCE_LENGTH]

        new_y.append(num_tag)

    # 将y中的元素编码成ont-hot encoding
    # ()
    y = np.empty(shape=(len(tags), MAX_SEQUENCE_LENGTH, len(label_id_dict.keys())+1))

    for i, seq in enumerate(new_y):
        y[i, :, :] = to_categorical(seq, num_classes=len(label_id_dict.keys())+1)
    return y


train_outputs = compute_output_arrays(train_tags)
test_outputs = compute_output_arrays(test_tags)
# print('outputs',outputs)
# y = np.array(to_categorical(outputs))


# %%

def create_model(n_tags):
    input_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    input_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    input_atn = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    config = BertConfig.from_pretrained(BERT_PATH + 'config.json')
    config.output_hidden_states = False
    bert_model = TFBertModel.from_pretrained(
        BERT_PATH + 'tf_model.h5', from_pt=False, config=config)
    # if config.output_hidden_states = True, obtain hidden states via bert_model(...)[-1]
    embedding = bert_model(input_id, attention_mask=input_mask, token_type_ids=input_atn)[0]
    # embedding:(None, 128, 768)
    lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=128, return_sequences=True), name="bi_lstm")(embedding)
    # print('lstm', lstm.shape)  # (None, 128, 256)
    drop = tf.keras.layers.Dropout(0.1)(lstm)
    # n_tags==5

    dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_tags, activation="softmax"), name="time_distributed")(drop)
    # print('dense',dense.shape) # (None, 128, 6)
    CRF = ConditionalRandomField(lr_multiplier=crf_lr_multiplier)
    output = CRF(dense)
    # output = tf.keras.layers.Lambda(lambda x: x ** 2)(dense)
    # print('output', output.shape)  # (None, 128, 6)
    model = tf.keras.models.Model(inputs=[input_id, input_mask, input_atn], outputs=output)
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(loss=CRF.sparse_loss, optimizer=optimizer, metrics=[CRF.sparse_accuracy])

    return model


# %%

print('len(inputs)',len(inputs))  # 3
# print('inputs.shape',inputs.shape)
# batch_size_per_replica = 64
# strategy = tf.distribute.MirroredStrategy()
# print('Number of devices: %d' % strategy.num_replicas_in_sync)  # 输出设备数量
# batch_size = batch_size_per_replica * strategy.num_replicas_in_sync
if __name__ == '__main__':

    # with strategy.scope():
    model = create_model(6)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
    # model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc', 'mae'])
    model.fit(inputs, train_outputs, validation_data=[test_inputs, test_outputs],
              epochs=1, batch_size=32)
    model.save_weights("%s_ner.h5" % "ccks2020")


