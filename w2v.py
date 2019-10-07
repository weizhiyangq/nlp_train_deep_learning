# -*- coding: utf-8 -*-
"""
Created on Tue May 14 22:28:08 2019

@author: YWZQ
"""

import logging
import os
from gensim.models import word2vec
import numpy as np
import jieba
from scipy.linalg import norm
import re

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

'''
with open('../corpus_cutfile/holiday_segment.txt', mode='r', encoding='gbk') as f:
    holiday_segment = f.readlines()
    print(holiday_segment[:10])
    print(len(holiday_segment))
    
with open('../corpus_cutfile/in_the_name_of_people_segment.txt', mode='r', encoding='gbk') as f:
    in_the_name_of_people_segment = f.readlines()
    print(in_the_name_of_people_segment[:10])
    print(len(in_the_name_of_people_segment))
    
with open('../corpus_cutfile/law_segment.txt', mode='r', encoding='gbk') as f:
    law_segment = f.readlines()
    print(law_segment[:10])
    print(len(law_segment))
    
with open('../corpus_cutfile/shopping_comment_segment.txt', mode='r', encoding='gbk') as f:
    shopping_comment_segment = f.readlines()
    print(shopping_comment_segment[:10])
    print(len(shopping_comment_segment))
    
with open('../corpus_cutfile/yitian_segment.txt', mode='r', encoding='gbk') as f:
    yitian_segment = f.readlines()
    print(yitian_segment[:10])
    print(len(yitian_segment))
    
all_segment = holiday_segment+in_the_name_of_people_segment+law_segment+shopping_comment_segment+yitian_segment
all_segment = [re.sub('\n','',i)for i in all_segment]
np.savetxt('../corpus_cutfile/all_segment.txt',all_segment,fmt='%s',encoding='utf-8')
all_segment = [i.split(' ') for i in all_segment]
print(len(all_segment))
'''


'''
#word2vec.Word2Vec(sentences, hs=1,min_count=1,window=3,size=100) 中的senctence可以是直接输入一个文本，用 word2vec.LineSentence读取，也可以是一个列表，列表包含所有文章，然后列表中每个元素也是一个列表，包含句子的分词
model = word2vec.Word2Vec(all_segment, hs=1,min_count=1,window=3,size=100)  
model.save('w2v_of_muti_txt.model')
model.wv.save_word2vec_format('w2v_of_muti_txt.txt',fvocab='w2v_vocab_of_multi_txt.txt')
#w2v_of_muti_txt.txt为词向量，每行为对应的词以及向量，在语料库中出现频次越多的越靠前。fvocab则是词以及出现的次数，也是频次越多越靠前

'''
sentences = word2vec.LineSentence('../corpus_cutfile/all_segment.txt')
model = word2vec.Word2Vec(sentences, hs=1,min_count=1,window=3,size=100) 
model.save('w2v_of_muti_txt2.model') 
model.wv.save_word2vec_format('w2v_of_muti_txt2.txt',fvocab='w2v_vocab_of_multi_txt2.txt')#w2v_of_muti_txt.txt为词向量，每行为对应的词以及向量，在语料库中出现频次越多的越靠前。fvocab则是词以及出现的次数，也是频次越多越靠前

##对应的加载方式
model= word2vec.Word2Vec.load("w2v_of_muti_txt2.model")
print('张无忌和张三丰的相似度：',model.similarity("张无忌", "张三丰"))
print('张无忌和周芷若的相似度：',model.similarity("张无忌", "周芷若"))
print('张无忌和赵敏的相似度：',model.similarity("张无忌", "赵敏"))
print('张无忌和成昆的相似度：',model.similarity("张无忌", "成昆"))
print('张无忌和高育良的相似度：',model.similarity("张无忌", "高育良"))
print('李达康和高育良的相似度：',model.similarity("李达康", "高育良"))
print('赵敏和周芷若的相似度：',model.similarity("赵敏", "周芷若"))
print('灭绝和周芷若的相似度：',model.similarity("灭绝师太", "周芷若"))
print('张无忌和殷素素的相似度：',model.similarity("张无忌", "殷素素"))
print('殷素素和张翠山的相似度：',model.similarity("张翠山", "殷素素"))
print('张无忌和酒店的相似度：',model.similarity("张无忌", "酒店"))
print('张无忌和少林的相似度：',model.similarity("张无忌", "少林"))
print('张无忌和武当的相似度：',model.similarity("张无忌", "武当"))
print('张无忌和明教的相似度：',model.similarity("张无忌", "明教"))
print('酒店和旅馆的相似度：',model.similarity("酒店", "旅馆"))
print('酒店和法院的相似度：',model.similarity("酒店", "法院"))
def vector_similarity(s1, s2):
    def sentence_vector(s):
        words = jieba.lcut(s)
        v = np.zeros(100)
        for word in words:
            v += model[word]
        v /= len(words)
        return v
    
    v1, v2 = sentence_vector(s1), sentence_vector(s2)
#    print('begin similar')
    return np.dot(v1, v2) / (norm(v1) * norm(v2))
s1="张无忌和赵敏去饭店"
s2="张无忌和赵敏去旅馆"
s3="张无忌和周芷若去明教"
s4="李达康去饭店"
s1_s2=vector_similarity(s1,s2)
s1_s3=vector_similarity(s1,s3)
s1_s4=vector_similarity(s1,s4)
print(s1_s2)
print(s1_s3)
print(s1_s4)