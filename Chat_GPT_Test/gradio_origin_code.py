import os
import openai
import gradio as gr

from pdf_split import summary
api_key = "sk-TzHeBadtGcaxJR6HrUxrHgygExhiTF2RLw6fudFM87tyhqeC"
base_url = "https://api.chatanywhere.tech/v1"
openai.api_key = api_key
openai.api_base = base_url

history_messages = []
def api_calling(input_text, history_conversation):
    if history_conversation:
        history_messages.extend([
			{"role": "user", "content": f"{history_conversation[-1][0]}"},
            {"role": "assistant", "content": f"{history_conversation[-1][1]}"}
		]
		)

    messages = history_messages+[
        {
            "role": "user",
            "content":  f"{input_text}"
        }
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0125",
        temperature=0.8,
        top_p=0.5,
        frequency_penalty=0,
        presence_penalty=0.8,
        messages=messages
    )

    return response
def message_and_history(input, history):
    history = history or []
    output = api_calling(input, history)
    history.append((input, str(output)))
    return history, history


block = gr.Blocks(theme=gr.themes.Monochrome())
with block:
    gr.Markdown("""<h1><center>🤖️对话机器人</center></h1>
    """)
    gr.Markdown("""<h2><center>🤖️对话机器人</center></h1>
        """)
    chatbot = gr.Chatbot()
    message = gr.Textbox()
    #file = gr.File(file_count="single")


    state = gr.State()

    submit = gr.Button("发送")
    generate = gr.Button("生成评价")

    submit.click(message_and_history,
          inputs=[message, state],
          outputs=[chatbot, state])

block.launch(share=True, debug=True)
