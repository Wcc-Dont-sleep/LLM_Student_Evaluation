import pandas as pd
import nltk
import re
from langchain_community.document_loaders.notebook import remove_newlines
from nltk import sent_tokenize
from pandas.core.computation.parsing import tokenize_string
import openai
from openai.embeddings_utils import distances_from_embeddings
from transformers import BertTokenizer
import os
api_key = "sk-Cd9qsD7Sjk5FLtBlb6fZ5GMRQmkmGfbZxZHxJPyBrIiFXU9V"
base_url = "https://api.chatanywhere.com.cn/v1"
os.environ["OPENAI_API_KEY"] = api_key
os.environ["OPENAI_API_BASE"] = base_url
openai.api_key = api_key
openai.api_base = base_url


# 初始化 BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

min_tokens = 5
max_tokens = 50

def text_cleaner(text):
    text = text.lower()
    text = text.replace('\n', ' ')
    text = text.replace('\t', ' ')
    text = text.replace('\r', ' ')
    text = text.replace('  ', ' ')
    text = text.replace('*', ' ')
    text = text.replace('#', ' ')
    text = text.replace('!', ' ')
    text = text.replace('-', ' ')
    text = text.replace('<', ' ')
    text = text.replace('>', ' ')
    text = re.sub(r'\s+', ' ', text)

    for char in text:
        # print("英文")
        en_sentences = sent_tokenize(text)
        print('英文书籍分句结果：', en_sentences)
        return en_sentences


def split_token(sentences, min_tokens=min_tokens, max_tokens=max_tokens):
    # Split the text into sentences


    # Get the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]
    print(f"n_tokens: {n_tokens}")
    chunks = []
    tokens_so_far = 0
    chunk = []

    for sentence, token in zip(sentences, n_tokens):
        if token<=1:
            continue
        print(sentence)
        print(token)

        # Check if adding this sentence would exceed the token limit
        if tokens_so_far + token > min_tokens:
            # If so, add the current chunk to the list of chunks and start a new chunk
            chunk.append(sentence)
            chunks.append(" ".join(chunk))
            chunk = []
            tokens_so_far = 0
            continue

        # Add the current sentence to the current chunk
        chunk.append(sentence)
        tokens_so_far += token

    if len(chunk)>1:
        chunks.append(" ".join(chunk))
    print("1-----------------------------分割后的字符串的临时列表")
    # 用于存放分割后的字符串的临时列表
    temp_list = []
    # 遍历原列表中的元素
    for item in chunks:
        # 使用tokenizer计算当前字符串的token数目
        tokens = tokenizer.encode(item)

        print(f"tokens: {len(tokens)}")
        print(tokens)
        print(100*'-')

        # 如果当前字符串的token数目不超过20，则直接将其添加到临时列表中
        if len(tokens) <= max_tokens:
            temp_list.append(item)
        else:
            # 否则，将当前字符串切分成若干个token数目不超过20的子串
            for sub_item in tokenize_string(item, max_tokens):
                temp_list.append(sub_item)

    # 将临时列表中的元素赋值给原列表
    my_list = temp_list
    return my_list
def tokenize_string(input_string, min_tokens):
    """
    将字符串切分成不超过 min_tokens 个 tokens 的子串。

    参数：
        input_string: 要切分的字符串。
        min_tokens: 每个子串的最小 tokens 数目。

    返回值：
        包含切分后子串的列表。
    """
    # 使用空格将字符串切分成单词列表
    words = input_string.split()

    # 初始化子串列表
    sub_strings = []

    # 初始化当前子串
    current_sub_string = ''

    # 遍历单词列表
    for word in input_string:
        # 如果当前子串的 tokens 数目加上当前单词的 tokens 数目不超过 min_tokens，则添加当前单词到当前子串中
        if len(current_sub_string.split()) + len(word.split()) <= min_tokens:
            current_sub_string += word + ' '
        # 否则，将当前子串添加到子串列表中，并初始化一个新的子串，将当前单词添加到新的子串中
        else:
            sub_strings.append(current_sub_string.strip())
            current_sub_string = word + ' '

    # 添加最后一个子串
    if current_sub_string:
        sub_strings.append(current_sub_string.strip())

    return sub_strings


with open('test_text.txt', 'r', encoding='utf-8') as file:
    # 读取文件内容
    text = file.read()
ln_sentences = text_cleaner(text)

df = pd.DataFrame(ln_sentences, columns=['text'])
#移除换行 空格
df['text'] = remove_newlines(df.text)

r=split_token(df['text'])
print(r)
df = pd.DataFrame(r, columns = ['text'])
df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
df.n_tokens.hist()
print(df)
model = "gpt-3.5-turbo"

# embedding the knowledge base
df['embeddings'] = df.text.apply(
    lambda text: openai.Embedding.create(input=text, engine='text-embedding-ada-002')['data'][0]['embedding'])

df.to_csv('processed/embeddings.csv')
df.head()



def create_context(question, df, max_len=1800):
    print("\n\n")
    print("questionSTART:\n" + question)
    print("\n\n")
    print("questionEND:\n")
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')

    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():

        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)

question = input()
context = create_context(question, df)

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    temperature=0,
    max_tokens=max_tokens,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    messages=[

        {"role": "user",
         "content": f"Answer the question based on the context below with English, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:"},
    ]
)
result = ''
for choice in response.choices:
    result += choice.message.content

print(result)

