import gradio as gr
import time
import threading


# 定义函数 A，接收一个参数
def A(param):
    time.sleep(2)  # 模拟一个需要2秒钟执行的任务
    return f"任务完成，参数为: {param}"


# 定义一个函数来计算 A 的执行时间并实时更新
def timed_A(param, elapsed_time_state):
    start_time = time.time()

    # 启动一个后台线程来实时更新执行时间
    def update_time():
        while True:
            elapsed_time = time.time() - start_time
            elapsed_time_state.value = elapsed_time
            time.sleep(0.1)  # 每0.1秒更新一次
            if elapsed_time >= 2:  # 假设任务最多运行2秒
                break

    update_thread = threading.Thread(target=update_time)
    update_thread.start()

    # 调用函数 A
    result = A(param)
    elapsed_time = time.time() - start_time
    return result, f"总执行时间: {elapsed_time:.2f} 秒"


# 创建 Gradio 接口
with gr.Blocks() as demo:
    gr.Markdown("### 实时更新函数执行时间的示例")

    # 输入框用于输入参数
    param_input = gr.Textbox(label="输入参数")

    # 创建一个按钮和输出区域
    with gr.Row():
        btn = gr.Button("执行任务")
        output_text = gr.Textbox(label="任务结果")
        time_text = gr.Textbox(label="执行时间")

    # 创建一个状态用于实时更新执行时间
    elapsed_time_state = gr.State(0)


    def on_click(param):
        result, total_time = timed_A(param, elapsed_time_state)
        return result, total_time


    # 定义一个函数来更新显示的执行时间
    def update_time_text():
        return f"执行时间: {elapsed_time_state.value:.2f} 秒"


    # 将按钮点击事件绑定到 timed_A 函数
    btn.click(on_click, inputs=param_input, outputs=[output_text, time_text])

    # 定期更新执行时间显示
    demo.load(lambda: None, [], time_text, every=0.1,fn=update_time_text)

# 启动 Gradio 应用
demo.launch()
