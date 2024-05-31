import openai

api_key = "sk-Cd9qsD7Sjk5FLtBlb6fZ5GMRQmkmGfbZxZHxJPyBrIiFXU9V"
base_url = "https://api.chatanywhere.com.cn/v1"

openai.api_key = api_key
openai.api_base = base_url

prompt = '''你是一个软件工程专业领域的文章归纳器。请根据下述对于学生软件设计文档的知识能力总结进行归纳,请注意：
            1.请根据总结中关于文档所体现的学生知识能力和学生知识缺陷分别生成归纳内容;
            2.请删除知识能力归纳内容和缺陷归纳内容中重复的部分;
            3.请根据缺陷归纳内容，生成对该学生下一步学习的建议，请使用如下模板"对该生进一步的学习建议为：";
            4.生成的下一步学习建议尽量详细;
            5.以下为需要进行归纳的内容：'''

file_path = "./processed/sorted_ans1.txt"

with open(file_path, 'rb') as file:
    text = file.read()

content = prompt+str(text)

response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        max_tokens=4000,
        temperature=0.8,
        top_p=0.5,
        frequency_penalty=0,
        presence_penalty=0,
        messages=[

            {"role": "user",
             "content": content},
        ]
    )

result = ''
for choice in response.choices:
    result += choice.message.content

print(result)
