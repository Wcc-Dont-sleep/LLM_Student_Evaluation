#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author : andy
@Date   : 2023/8/23 10:09
@Contact: 864934027@qq.com
@File   : chunk_summary.py
"""
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
def callChatgpt(system_prompt,user_prompt):
    if system_prompt == "":
        message = [
            {"role": "user",
             "content": user_prompt},
        ]
    else:
        message = [
            {"role": "system",
             "content": system_prompt},

            {"role": "user",
             "content": user_prompt},
        ]
    for i in range(10):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                temperature=0.8,
                top_p=0.5,
                frequency_penalty=0,
                presence_penalty=0.8,
                messages=message
            )
            return response
        except Exception as e:
            print(f"第{i + 1}次尝试失败...")
def callChatGLM6B(prompt):
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
    # chunks
    prompt_pre = '''你是一个软件工程领域的知识点提取器，请提取下述内容中存在的软件工程领域知识点。请注意：
                    1.请不要直接获取原文中的段落标题部分，请根据段落内容概括所体现的知识点。且生成内容只包含知识点名称，不要展开。
                    2.请尽量概括提取2-3个知识点，要求符合原段落整体所展现的软件工程知识。
                    3.知识点实例：‘系统功能性需求’，‘安全性’，‘鲁棒性’，‘系统架构’等
                    4.对下述内容进行知识点提取：'''

    s_prompt0 = '''你是一个软件工程专业知识领域的摘要生成器,你存在如下背景知识：'''
    prompt1 = '''请参考提供的背景知识对如下内容进行分段总结,请注意：
                   1.输入数据为从pdf读入的学生设计的软件的文档，请先删除参考文献相关内容;
                   2.要求每段内容不丢失主要信息, 每段的字数在100字左右;
                   3.生成的总结内容为该段落所体现的软件工程知识能力总结，能力总结包括段落所涵盖的软件工程知识和段落中体现软件工程知识缺陷的部分;
                   4.每段的摘要请将所有展现的知识点自然链接，且不要出现重复的知识点概述。
                   5.摘要中对展现知识的描述请不要只是描述知识名词，而要具体的阐述相关知识内容。
                   6.每段生成的摘要开头一定不要含有'该段落'的前缀文字;
                   7.对下文进行分段总结:'''


    prompt3 = '''你是一个文章内容整合器，请注意：
                1.输入数据中含有多个已经总结好的段落;
                2.请不要在整合内容中出现“该段落”前缀文字。
                3.请将每段信息进行优化,使得每段连接显得更加连贯,且保留每段的大部分信息;
                4.生成的整合内容连贯、完整，形成一个完整的段落内容。
                5.输入的的文章如下：'''

    prompt4 = '''你是一个软件工程专业领域的文章归纳器。请根据下述对于学生软件设计文档的知识能力总结进行归纳,请注意：  
                1.请严格根据归纳内容全文生成三个部分内容：1.学生知识能力；2.学生知识缺陷；3.进一步学习建议;并保证生成所有内容符合软件工程规范。  
                2.生成内容尽量详细，如缺陷部分详细描述在原文档中哪部分内容体现了该缺陷。
                3.请注意，归纳内容中提及到的已经存在的知识内容请不要出现在学生知识缺陷中;  
                4.请检查三个部分的所有内容是否都符合归纳内容全文的意思。  
                5.请保证归纳内容中所有的缺陷描述都被归纳进“学生知识缺陷”部分;
                6.请对输出内容进行分点描述。
                7.以下为需要进行归纳的内容：'''
    ans = ""
    for i in range(len(chunks)):
        if len(chunks[i]) < 50:
            continue
        knowledge_ans = ''
        # response = callChatGLM66B(prompt + chunks[i])
        response = callChatgpt("", prompt_pre + chunks[i])
        for choice in response.choices:
            knowledge_ans += choice.message.content

        bg_text = search_emb_db(knowledge_ans)
        response = callChatgpt(s_prompt0 + bg_text, prompt1 + chunks[i])

        #response = callChatGLM6B(prompt + chunks[i])
        if 'choices' not in response.keys():
            print(response.keys(), "\n")
            print("========this chunk has problem=======\n")
            continue
        temp_ans = ''
        for choice in response.choices:
            temp_ans += choice.message.content
        temp_ans = temp_ans + "\n"
        # print(temp_ans)
        # temp_ans = response['choices'][0]['content'] + "\n"
        ans += temp_ans

    ans = ans.replace("\\n", '\n')

    """
    response = callChatGLM6B(prompt3+ans)
    sorted_ans = ''
    for choice in response.choices:
        sorted_ans += choice.message.content
    """
    response = callChatgpt("", prompt4 + ans)
    classifier_ans = ''
    for choice in response.choices:
        classifier_ans += choice.message.content

    # save txt
    # save_path = "/Users/guomiansheng/Desktop/LLM/LangChain-ChatLLM/save_6b_ans3_all"
    save_path = "./Gradio_processed"
    with open(save_path + '/ans' + str(num) + '.txt', 'w', encoding='utf-8') as file:
        file.write(ans)
    print("======ans========:\n", ans)


    # with open(save_path + '/sorted_ans' + str(num) + '.txt', 'w', encoding='utf-8') as file:
        # file.write(sorted_ans)
    # print("======sorted_ans========:\n", sorted_ans)

    with open(save_path + '/classifier_ans' + str(num) + '.txt', 'w', encoding='utf-8') as file:
        file.write(classifier_ans)
    print("======classifier_ans========:\n", classifier_ans)

    one_dict = {'input': page_ans, "output": classifier_ans}
    return classifier_ans, one_dict

#summary("./data/pdf_doc/courseproject1_233318_694182_SAD_assignment_1.pdf",13)
#print("1")
def final_evaluation(message):
    sys_prompt = '''你是一个软件工程专业领域的课程作业评价生成器'''
    prompt = '''请根据下述对于多份作业评价生成对该学生能力成长的总评,请注意：
                1.内容来自同一个学生的不同作业的评价集合。每份评价都包含1.学生知识能力；2.学生知识缺陷；3.进一步学习建议三个部分。
                2.生成评价模板如下：“该生在（1）方面仍有缺陷，具体问题为：（2）；\n该生已经掌握了（3）方面知识点；\n该生在（4）方面有所改进，具体表现为：（5）；\n该生在（6）方面有所退步，具体表现为：。”
                3.请不要对每一份评价都生成一次模板内容，而要根据所有评价生成一份评价
                4.考虑每一份评价，将出现在“学生知识缺陷”中未出现在“学生知识能力”中的部分放入（2）中，并提取其中包含的软件工程知识点名称放入（1）中；  
                5.考虑所有评价，将所有出现在“学生知识能力”中但未出现在“学生知识缺陷”中的内容，提取其中包含的软件工程知识点名称放入（3）中。
                6.考虑所有评价，将所有曾经出现在“学生知识缺陷”中但后续出现在“学生知识能力”中的内容放入（5）中，并提取其中包含的软件工程知识点名称放入（4）中。如果没有符合要求的内容，则删去“该生在（4）方面有所改进，具体表现为：（5）”这句话。
                7.考虑所有评价，将所有曾经出现在“学生知识能力”中但后续出现在“学生知识缺陷”中的内容，提取其中包含的软件工程知识点名称放入（6）中。如果没有符合要求的内容，则删去“该生在（6）方面有所退步”这句话；
                8.请保证最后的生成内容中只包含一份模板内容。
                9.以下为多份评价：'''
    prompt1 = '''请根据下述对于多份作业评价生成对该学生能力成长的总评,请注意：
                    1.内容来自同一个学生的不同作业的评价集合。每份评价都包含1.学生知识能力；2.学生知识缺陷；3.进一步学习建议三个部分。
                    2.评价集合内容是按照时间顺序排序的。
                    3.请按照顺序考虑所有评价内容，评价学生在哪些方面有所进步，在那些方面有所退步，请保留原评价集合中的主要内容，所有内容请分点回答。
                    4.请汇总进一步学习建议，删除重复部分，放在评价最后，命名为：“对该生的进一步学习建议如下”。
                    5.以下为多份评价：'''
    response = callChatgpt(sys_prompt, prompt1 + message)

    if 'choices' not in response.keys():
        print(response.keys(), "\n")
        print("========Evaluation set have problem=======\n")

    temp_ans = ''
    for choice in response.choices:
        temp_ans += choice.message.content

    return temp_ans

def test_single():
    classifier_ans_list = ""
    for i in range(3):
        file_name = "classifier_ans" + str(i) + ".txt"
        with open('./Gradio_processed/'+file_name, 'r', encoding='utf-8') as f:
            res = f.read()
            classifier_ans_list += (f"第{i+1}份评价如下：{res}")

    res = final_evaluation(classifier_ans_list)
    save_path = "./Gradio_processed"
    num = 1
    with open(save_path + '/final_ans_direct' + str(num) + '.txt', 'w', encoding='utf-8') as file:
        file.write(res)
    print("======final evaluation========:\n", res)

    return res
def single_evaluation(folder_path):
    classifier_ans_list = ""
    with zipfile.ZipFile(folder_path, 'r') as zip_ref:
        # 获取所有文件的列表
        file_list = zip_ref.namelist()

        # 遍历每个文件并执行 summary 函数
        for i, file_name in enumerate(file_list):
            with zip_ref.open(file_name) as file:
                # file_content = file.read().decode('utf-8')
                res,_ = summary(file, i)
                classifier_ans_list += (f"第{i+1}份评价如下：{res}")

        res = final_evaluation(classifier_ans_list)
        save_path = "./Gradio_processed"
        num = 1
        with open(save_path + '/final_ans' + str(num) + '.txt', 'w', encoding='utf-8') as file:
            file.write(res)
        print("======final evaluation========:\n", res)
    return res

#single_evaluation("./data/pdf_doc/Test_data.zip")
# test_single()
print(1)

"""
def main():
    # find 10 file
    def find_files_with_prefix(folder_path, prefix):
        matching_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.startswith(prefix) and file.endswith('.pdf'):
                    matching_files.append(os.path.join(root, file))
        return matching_files

    # 示例用法
    folder_path = './data/pdf_doc'  # 替换为你的大文件夹路径
    # prefixes = ['pdf_0', 'pdf_1', 'pdf_2']  # 替换为你想要匹配的前缀列表
    prefixes = []
    for i in range(10):
        prefixes.append('pdf_' + str(i))
    matching_files = []
    for prefix in prefixes:
        matching_files.extend(find_files_with_prefix(folder_path, prefix))
    # del matching_files[0]
    # del matching_files[0]

    ans_lst = []
    for i in range(len(matching_files)):
        one_ans, one_dict = summary(matching_files[i], i)
        ans_lst.append(one_dict)
    # pdf_path = "/Users/guomiansheng/Desktop/LLM/LangChain-ChatLLM/pdf_test.pdf"
    # summary(pdf_path)
    return ans_lst


def preprocess_data(ans_lst):
    json_path = "./processed/pdf_doc.json"
    with open(json_path, "w", encoding='utf-8') as fout:
        for dct in ans_lst:
            line = json.dumps(dct, ensure_ascii=False)
            fout.write(line)
            fout.write("\n")

"""
def read_data():
    json_path = "./processed/pdf_doc.json"
    with open(json_path, "r", encodings='utf-8') as f:
        lst = [json.loads(line) for line in f]
        df = pd.json_normalize(lst)
        print(df.head())



