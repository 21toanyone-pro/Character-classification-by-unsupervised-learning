# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 18:10:20 2019

@author: Lim
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import pandas as pd
from pandas import DataFrame
import operator



#글자 사전(사용된건지 아닌건지 테스트)
with open('chin.txt', 'r', encoding='UTF8') as file_object:
    contents = file_object.read() #단어 저장
    
#검사할 txt 파일
#'C:/Users/DUSK/Desktop/git/Text_Crawling/전체실록/선조실록.txt'
document_text = open('1페이지.txt', 'r', encoding='UTF8' )
text_string = document_text.read()

#갯수 저장
save = open('조선왕조실록카운트.txt', 'w', encoding='utf-8')

save_chin = open('조선왕조실록사용글자.txt', 'w', encoding='utf-8')
#없는 글자
save_non = open('없는 글자.txt', 'w', encoding='utf-8')

#보기 편하게 10글자 마다 자름
num = 0
#총 글자 갯수
lengths = 0
print('hi)')
#사용 안된 글자 카운트
zeros =0
data = []
inputs = []
Dicts = {}
dic_count = len(text_string)
Big_test = [10]



#검사
for i in range(len(contents)):#contents
    counting = str(text_string.count(contents[i])) #갯수 세는 거
    
    #print(un_countiong)
    lengths = lengths + text_string.count(contents[i]) #총 길이
    #전체 길이 체크
    if text_string.count(contents[i]) !=0:  #들어간 글자 체크
        num = num +1
        Dicts[contents[i]] = int(counting) #딕
    
    if text_string.count(contents[i]) == 0: #안들어간 글자 체크
        zeros = zeros + 1
        data.append(contents[i])


for j in range(dic_count):  #안들어간 글자들 저장
    un_countiong = str(contents.count(text_string[j]))
    if contents.count(text_string[j])==0:
        save_non.write(text_string[j])



sorted_dic = sorted(Dicts.items(),key=(lambda x: x[1]),reverse= True) #딕셔너리 셔플
print(sorted_dic[:10])

for i in sorted_dic:
    save.write(i[0]+':'+str(i[1]) + '\n')
    save_chin.write(i[0])
    

#save.write('\n'+str(data))
save.write('\n'+'총:'+str(lengths)+'\n')
save.write('사용 글자 수:'+str(num)+'\n')
save.write('\n'+'안들어간 글자:'+str(zeros)+'\n')
save.write('평균:'+str(lengths / len(contents)))
#setting.good()


