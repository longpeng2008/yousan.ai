#coding=utf-8
import random, nltk
import jieba.posseg as pesg

stop_word = ['吗','呢','了','啦','哈','!','，','？','。',"？",'！',',','.']

#1 进行数据读取
def read_data(filename):
    data = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            (label, sentences) = line.split('\t')
            sentence_list = sentences.split('；')
            data.extend([(sentence,label) for sentence in sentence_list if sentence])
            # 最后返回的是一个列表，结构如下[('我要打车','get_a_taxi')，('明天天气怎么样','get_weather')。。。]
    return data

#2.1 停用词处理
def delte_stop_word(sentence):
    for word in stop_word:
        if word in sentence:
            sentence.replace(word, '')
    return sentence

#2 进行特征选择，这里利用分词后的词性作为特征
def get_word_features(sentence):
    data = {}
    sentence = delte_stop_word(sentence)
    seg_list = pesg.cut(sentence)
    for word, tag in seg_list:
        data[tag] = word
    return data


#3 构建训练数据集
def get_features_sets(datafile):
    feature_sets = []
    for sentence, label in read_data(datafile):
        feature = get_word_features(sentence)
        feature_sets.append((feature, label))
    return feature_sets


classifier = nltk.NaiveBayesClassifier.train( get_features_sets('data.txt'))
predict_label = classifier.classify(get_word_features('请问明天的天气怎么样？'))
print(predict_label)    ## get_weather
print(classifier.prob_classify(get_word_features('请问明天的天气怎么样？')).prob(predict_label)) # 0.995154

while True:
    print("请输入您要预测的句子：")
    sentente = input()
    predict_label = classifier.classify(get_word_features(sentente))

    prob = (classifier.prob_classify(get_word_features(sentente)).prob(predict_label))  # 0.995154
    print("文本<%s>预测类别为：%s 概率为 %f "%( sentente, predict_label, prob) )