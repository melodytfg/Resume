# Resume
import streamlit as st
import nltk
import jieba
import jieba.analyse
import requests
import spacy
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from spacy.lang.zh import Chinese
from spacy.lang.zh.stop_words import STOP_WORDS
from spacy import displacy
from spacy.tokens import Span
from collections import Counter
from heapq import nlargest
import re
import jieba.analyse
import jieba.posseg as pseg
from textblob import TextBlob

from spacy.tokens import Token
Token.set_extension('highlight_color', default=None, force=True)
from spacy.tokens import Doc, Span
Doc.set_extension('highlight_color', default=None, force=True)
Span.set_extension('highlight_color', default=None, force=True)


# 下載必要的 NLTK 資源
nltk.download('punkt')
nltk.download('stopwords')

# 定義應用程序的標題
st.title("履歷優化｜讓你的履歷更上一層樓！")

# 添加一個文本框
user_input = st.text_input("請輸入你的姓名：")

# 使用 text_input 函数创建一个输入框
st.subheader('請問你想去什麼公司？')
url = st.text_input('請輸入網址', '')

# 显示输入框的值
st.write('您輸入的網址是：', url)

# 添加一個選擇框
st.subheader('請問你想申請什麼職位？')
#job_type = st.selectbox("選擇您想申請的職位種類：",["軟體工程師", "設備工程師","數據分析師", "產品經理","行銷企劃","會計","行政人員","其他"])
job_type =[" ","軟體工程師", "設備工程師","數據分析師", "產品經理","行銷企劃","會計","行政人員","其他"]
# 如果用戶選擇“其他”，顯示一個文本輸入框
selected_option = st.selectbox('請選擇一個選項：', job_type)
if selected_option == '其他':
    other_option = st.text_input('請輸入其他選項：')
    if other_option:
        st.write('您輸入的是：', other_option)


# 加载中文语言模型
nlp = spacy.load('zh_core_web_sm')

# TextRank算法关键词提取函数
def extract_keywords(text):
    # 定义停用词列表
    stopwords = list(STOP_WORDS)
    # 去除空格和标点符号
    text = re.sub(r'[^\w\s]','',text)
    # 创建一个Spacy Doc对象
    doc = nlp(text)
    # 註冊 highlight_color 屬性
    doc.user_data["title"] = "Document title"
    Span.set_extension("highlight_color", default=None, force=True)
    # 创建一个单词列表
    words = [token.text for token in doc]
    # 计算单词频率并标准化
    word_freq = Counter(words)
    max_freq = max(word_freq.values())
    for word in word_freq.keys():
        word_freq[word] = (word_freq[word]/max_freq)
    # 计算句子权重
    sent_list = [sent for sent in doc.sents]
    sent_scores = {}
    for sent in sent_list:
        for word in sent:
            if word.text.lower() in word_freq.keys():
                if sent not in sent_scores.keys():
                    sent_scores[sent] = word_freq[word.text.lower()]
                else:
                    sent_scores[sent] += word_freq[word.text.lower()]
    # 提取关键句子
    summary_sents = nlargest(5, sent_scores, key=sent_scores.get)
    summary = [sent.text for sent in summary_sents]
    # 提取关键词
    keywords = nlargest(10, word_freq, key=word_freq.get)
    return summary, keywords

# 展示优化后的文本
def show_optimized_text(text, keywords):
    # 创建一个Spacy Doc对象
    doc = nlp(text)
    # 标记关键词
    for token in doc:
        if token.text in keywords:
            token._.set("highlight_color", "#ffff00")
    # 使用Displacy渲染文本
    html = displacy.render(doc, style="ent", options={"compact": True})
    # 显示渲染后的HTML
    st.write(html, unsafe_allow_html=True)


# Streamlit应用
def main():
    # 显示页面标题
    st.subheader("關鍵字提取與履歷優化")
    # 创建一个文本输入框，获取用户输入的文本
    text = st.text_area("請在此輸入履歷", height=250)
    # 创建一个按钮，让用户点击来触发优化操作
    if st.button("關鍵字提取"):
        # 提取关键句子和关键词
        summary, keywords = extract_keywords(text)
        # 显示关键句子
        st.subheader("關鍵句子：")
        for sent in summary:
            st.write(sent)
        # 显示关键词
        st.subheader("關鍵字：")
        st.write(keywords)
        # 显示优化后的文本
        show_optimized_text(text, keywords)

if __name__ == "__main__":
    main()

#完整度分析
import streamlit as st
from gensim.summarization import keywords

st.subheader('完整度分析')
resume = st.text_area('請在此輸入已優化過的履歷')

if st.button('完整度分析'):
    if resume.strip() == '':
        st.warning('請輸入履歷內容！')
    else:
        # 计算履历中的关键字
        keyword_list = keywords(resume, words=10, split=True, lemmatize=True)
        keyword_str = '、'.join(keyword_list)
        
        # 判断履历是否完整
        if len(keyword_list) >= 5:
            st.success(f'您的履歷完整度很高，其中關鍵字包括：{keyword_str}')
        else:
            st.error(f'您的履歷完整度較低，其中關鍵字包括：{keyword_str}')


#匹配度分析
import streamlit as st
import jieba.analyse
import numpy as np

def load_stopwords():
    stopwords_path = 'cnstopwords.txt'
    stopwords = set()
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        for line in f:
            stopwords.add(line.strip())
    return stopwords

def calculate_match_score(job_keywords, resume_keywords, stopwords):
    match_score = 0
    for word, weight in resume_keywords:
        if word in job_keywords and word not in stopwords:
            match_score += weight
    return match_score

st.subheader('匹配度分析')
    # 職位要求的關鍵字
job_requirement = st.text_input("請輸入職位要求的關鍵字（以空格分隔）：")
job_keywords = set(job_requirement.strip().split())

    # 中文履歷文本
resume_text = st.text_area("請輸入已優化過的履歷：")

    # 讀取停用詞
stopwords = load_stopwords()




# 按鈕觸發計算
if st.button("計算匹配度分數"):
    # 提取履歷文本的關鍵字
    resume_keywords = jieba.analyse.extract_tags(resume_text, topK=1000, withWeight=True, allowPOS=('n','v','vn','a','d'))

    # 計算匹配分數
    match_score = calculate_match_score(job_keywords, resume_keywords, stopwords)

    # 輸出匹配度分數
    max_score = np.sum([x[1] for x in resume_keywords])
    if max_score == 0:
        match_percentage = 0
    else:
        match_percentage = round(match_score / max_score * 100, 2)
    st.write(f"匹配度分數：{match_percentage}/100")
