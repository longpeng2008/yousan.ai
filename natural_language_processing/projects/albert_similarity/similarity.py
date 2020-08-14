#! -*- coding:utf-8 -*-
# 利用albert-tiny模型计算文本相似度，albert_zh权重下载(https://github.com/brightmart/albert_zh)

from bert4keras.utils import Tokenizer, load_vocab
from bert4keras.bert import build_bert_model
from keras.models import Model
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def similarity_count(vec1, vec2, mode='cos'):
    if mode == 'eu':
        return euclidean_distances([vec1,vec2])[0][1]
    if mode == 'cos':
        return cosine_similarity([vec1, vec2])[0][1]


maxlen = 128
config_path = 'albert_tiny_zh_google/albert_config_tiny_g.json'
checkpoint_path = 'albert_tiny_zh_google/albert_model.ckpt'
dict_path = 'albert_tiny_zh_google/vocab.txt'


tokenizer = Tokenizer(dict_path)

# 加载预训练模型
bert = build_bert_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    with_pool=True,
    albert=True,
    return_keras_model=False,
)

model = Model(bert.model.input, bert.model.output)

token_ids1, segment_ids1 = tokenizer.encode(u'我想去北京')
token_ids2, segment_ids2 = tokenizer.encode(u'我想去香港')
token_ids3, segment_ids3 = tokenizer.encode(u'目前的局势，止暴制乱，刻不容缓')

sentence_vec1 = model.predict([np.array([token_ids1]), np.array([segment_ids1])])[0]
sentence_vec2 = model.predict([np.array([token_ids2]), np.array([segment_ids2])])[0]
sentence_vec3 = model.predict([np.array([token_ids3]), np.array([segment_ids3])])[0]


print("《我想去北京》和《我想去香港》的余弦距离为%f"%similarity_count(sentence_vec1, sentence_vec2))
print("《我想去北京》和《我想去香港》的欧式距离为%f"%similarity_count(sentence_vec1, sentence_vec2, mode='eu'))

print("《我想去北京》和《目前的局势，止暴制乱，刻不容缓》的余弦距离为%f"%similarity_count(sentence_vec1, sentence_vec3))
print("《我想去北京》和《目前的局势，止暴制乱，刻不容缓》的欧式距离为%f"%similarity_count(sentence_vec1, sentence_vec3, mode='eu'))


