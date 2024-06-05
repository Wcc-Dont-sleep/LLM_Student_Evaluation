import json
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTTextBoxHorizontal
from pdfminer.pdfpage import PDFPage
import os
import pandas as pd
import openai
import zipfile
from chroma_test import search_emb_db
api_key = "sk-TzHeBadtGcaxJR6HrUxrHgygExhiTF2RLw6fudFM87tyhqeC"
base_url = "https://api.chatanywhere.tech/v1"

openai.api_key = api_key
openai.api_base = base_url

def split_document_by_page(pdf_file):
    resource_manager = PDFResourceManager()
    codec = 'utf-8'
    laparams = LAParams()
    device = PDFPageAggregator(resource_manager, laparams=laparams)
    interpreter = PDFPageInterpreter(resource_manager, device)

    split_pages = []

    #with open(pdf_path, 'rb') as file:
    for page in PDFPage.get_pages(pdf_file):
        interpreter.process_page(page)
        layout = device.get_result()
        text_blocks = []
        for element in layout:
            if isinstance(element, LTTextBoxHorizontal):
                text = element.get_text().strip()
                text_blocks.append(text)
        page_text = '\n'.join(text_blocks)
        split_pages.append(page_text)


    return split_pages
def callChatGLM6B(prompt):
    for i in range(10):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                temperature=0.8,
                top_p=0.5,
                frequency_penalty=0,
                presence_penalty=0.8,
                messages=[

                    {"role": "user",
                     "content": prompt},
                ]
            )
            return response
        except Exception as e:
            print(f"第{i + 1}次尝试失败...")

def summary(pdf_path, num):
    # 使用示例
    # pdf_path = "/Users/guomiansheng/Desktop/LLM/LangChain-ChatLLM/pdf_test.pdf"
    #pdf_path = "./data/pdf_doc/courseproject1_233318_694182_SAD_assignment_1.pdf"  # 替换为你的 PDF 文件路径
    one_dict = {}
    pages = split_document_by_page(pdf_path)
    add_page_data = ''
    page_ans = ""
    print(f"=============这是第{num}个pdf\n")
    for i, page_text in enumerate(pages):
        # page_ans = page_ans + f"这是第{i}页pdf:\n" + page_text
        page_ans = page_ans + page_text
        print(f"Page {i + 1}:", "当前page的字数:", len(page_text))
        print(page_text)
        print("--------------------")

    # 文本分片
    text_splitter = RecursiveCharacterTextSplitter(
        # separator="\n",
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_text(page_ans)
    prompt = '''请对下述学生作业生成评价。控制字数在100字以内'''


    ans = ""
    for i in range(len(chunks)):
        if len(chunks[i]) < 50:
            continue

        temp_ans = ''
        response = callChatGLM6B(prompt + chunks[i])
        if 'choices' not in response.keys():
            print(response.keys(), "\n")
            print("========this chunk has problem=======\n")
            continue
        for choice in response.choices:
            temp_ans += choice.message.content

        temp_ans = temp_ans + "\n"
        # print(temp_ans)
        # temp_ans = response['choices'][0]['content'] + "\n"
        ans += temp_ans
    ans = ans.replace("\\n", '\n')


    # save txt
    # save_path = "/Users/guomiansheng/Desktop/LLM/LangChain-ChatLLM/save_6b_ans3_all"
    save_path = "./Gradio_compared_processed"
    with open(save_path + '/ans' + str(num) + '.txt', 'w', encoding='utf-8') as file:
        file.write(ans)
    print("======ans========:\n", ans)


    # with open(save_path + '/sorted_ans' + str(num) + '.txt', 'w', encoding='utf-8') as file:
        # file.write(sorted_ans)
    # print("======sorted_ans========:\n", sorted_ans)
    return ans

prompt_final = '''请总结下述学生作业评价'''
def single_evaluation(folder_path):
    classifier_ans_list = ""
    with zipfile.ZipFile(folder_path, 'r') as zip_ref:
        # 获取所有文件的列表
        file_list = zip_ref.namelist()

        # 遍历每个文件并执行 summary 函数
        for i, file_name in enumerate(file_list):
            with zip_ref.open(file_name) as file:
                # file_content = file.read().decode('utf-8')
                res = summary(file, i)
                classifier_ans_list += (f"第{i+1}份评价如下：{res}")

        response = callChatGLM6B(prompt_final + classifier_ans_list)
        res = ''
        for choice in response.choices:
            res += choice.message.content

        save_path = "./Gradio_compared_processed"
        num = 0
        with open(save_path + '/final_ans' + str(num) + '.txt', 'w', encoding='utf-8') as file:
            file.write(res)
        print("======final evaluation========:\n", res)
    return res

single_evaluation("./data/pdf_doc/Test_data.zip")