
from openai import OpenAI

api_key = "sk-Cd9qsD7Sjk5FLtBlb6fZ5GMRQmkmGfbZxZHxJPyBrIiFXU9V"
base_url = "https://api.chatanywhere.cn/v1"

client = OpenAI(
    api_key=api_key,
    base_url=base_url
)


def get_completion(prompt,story, model="gpt-3.5-turbo-0125"):
    messages = [
        {
            "role": "user",
            "content": story
        },
        {
            "role": "system",
            "content": prompt
        }
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,

    )
    return response.choices[0].message.content


file_path = "story.txt"
story = ""
with open(file_path, encoding='utf-8') as file:
    # 读取文件内容
    story = file.read()

prompt = "请告诉我这个故事的中心思想是什么？"
print(get_completion(prompt, story))



