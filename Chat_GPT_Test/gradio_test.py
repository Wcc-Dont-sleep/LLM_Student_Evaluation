import os
import zipfile
from datetime import time
import gradio as gr
import time
from pdf_split import summary,final_evaluation,single_evaluation
api_key = "sk-TzHeBadtGcaxJR6HrUxrHgygExhiTF2RLw6fudFM87tyhqeC"
base_url = "https://api.chatanywhere.tech/v1"
def generate_evaluation(file_path,progress=gr.Progress()):
    start_time = time.time()
    classifier_ans_list = ""
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        # 获取所有文件的列表
        file_list = zip_ref.namelist()

        # 遍历每个文件并执行 summary 函数
        for i in progress.tqdm(range(len(file_list))):
            file_name = file_list[i]
            with zip_ref.open(file_name) as file:
                time.sleep(2)
                res, _ = summary(file, i+9)

                classifier_ans_list += (f"第{i + 1}份评价如下：{res}")

        res = final_evaluation(classifier_ans_list)

        save_path = "./Gradio_processed"
        num = 0
        with open(save_path + '/final_ans' + str(num) + '.txt', 'w', encoding='utf-8') as file:
            file.write(res)
        print("======final evaluation========:\n", res)

    end_time = time.time()

    return res,end_time-start_time

def generate_evaluation_1(file_path):
    start_time = time.time()
    output = single_evaluation(file_path)
    end_time = time.time()
    return output, end_time-start_time
block = gr.Blocks(theme=gr.themes.Monochrome())
with block:
    gr.Markdown("""<h1><center>课程交付物智能分析系统</center></h1>
    """)
    gr.Markdown("""<h2><center>请上传一份包含有学生系列课程项目文档的ZIP格式文件</center></h1>
        """)
    #chatbot = gr.Chatbot()
    #message = gr.Textbox(placeholder="请提交你的文件")
    file = gr.File(file_count="single")
    time_text = gr.Textbox(label="总用时",placeholder="评价结束后将展示总用时")
    evaluation = gr.Textbox(label="学生个人能力成长评价")


    #submit = gr.Button("发送")
    generate = gr.Button("生成评价")
    generate.click(generate_evaluation,
          inputs=[file],
          outputs=[evaluation, time_text])

    """
    submit.click(message_and_history,
          inputs=[message, state, file],
          outputs=[chatbot, state])
    """
block.launch(share=True, debug=True)
