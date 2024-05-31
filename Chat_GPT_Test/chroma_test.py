import chromadb
import os
import re
from chromadb.utils import embedding_functions
import openai
api_key = "sk-TzHeBadtGcaxJR6HrUxrHgygExhiTF2RLw6fudFM87tyhqeC"
base_url = "https://api.chatanywhere.tech/v1"
openai.api_key = api_key
openai.api_base = base_url

chroma_client = chromadb.Client()
client = chromadb.PersistentClient(path="./Database")
doc_list = []
doc_meta_list = []
doc_id_list = []
folder_path = "./data/Test"

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key,
                api_base=base_url,
                model_name="text-embedding-ada-002"
            )

collection = chroma_client.get_or_create_collection("Knowledge_base",embedding_function=openai_ef)

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

    return text

def chroma_initial(folder_path):
    i = 1
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                clean_content = text_cleaner(content)
                num_chunks = (len(clean_content) + 4999) // 5000  # 计算切割后的段落数

                # 将文本切割成每5000个字符一段
                for j in range(num_chunks):
                    start_idx = j * 5000
                    end_idx = (j + 1) * 5000
                    chunk = clean_content[start_idx:end_idx]

                    # 添加到doc_list
                    doc_list.append(chunk)

                    # 构建对应的doc_meta
                    doc_meta = {"Title": filename, "Chapter": j + 1}
                    doc_meta_list.append(doc_meta)

                    # 构建对应的doc_id
                    doc_id_list.append("id" + str(i))

                    i += 1

    """
    len_list = [len(x) for x in doc_list]
    print(len_list)
    """

    collection.add(
        documents=doc_list,
        metadatas=doc_meta_list,
        ids=doc_id_list
    )
def text_embedding(text) -> None:
    for i in range(10):
        try:
            response = openai.Embedding.create(model="text-embedding-ada-002", input=text)
            return response["data"][0]["embedding"]
        except Exception as e:
            print(f"第{i + 1}次尝试失败...")

# 初始化Chroma数据库
def search_emb_db(query):
    def is_folder_empty(folder_path):
        return len(os.listdir(folder_path)) == 0

    folder_path = './Database'
    if is_folder_empty(folder_path):
        chroma_initial(folder_path)

    query_embs = text_embedding(query)
    results = collection.query(

        query_embeddings=query_embs,
        # include=["documents"],
        n_results=2
    )
    res = "\n".join(str(item) for item in results['documents'][0])
    return res







"""
result structure is as below:
"ids":
"distance":
"metadatas":
"embeddings":
"documents": original text
"""