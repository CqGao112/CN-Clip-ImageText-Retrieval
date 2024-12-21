import gradio as gr

from model_loader import clip_text2image


clip_base = "中文CLIP(Base)"
clip_large = "中文CLIP(Large)"
clip_large_336 = "中文CLIP(Large,336分辨率)"
yes = "是"
no = "否"


def text2image_gr():
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Column(scale=2):
                    text = gr.Textbox(value="圣诞节快乐", label="请填写文本", elem_id=0, interactive=True)
                num = gr.components.Slider(minimum=0, maximum=50, step=1, value=8, label="返回图片数（可能被过滤部分）", elem_id=2)
                model = gr.components.Radio(label="模型选择", choices=[clip_base, clip_large, clip_large_336],
                                            value=clip_base, elem_id=3)
                # model = 'clip_base'
                thumbnail = gr.components.Radio(label="是否返回缩略图", choices=[yes, no],
                                                value=yes, elem_id=4)
                btn = gr.Button("搜索")
            with gr.Column(scale=100):
                out = gr.Gallery(label="检索结果为：",columns=4, height=450)
        inputs = [text, num, model, thumbnail]
        btn.click(fn=clip_text2image, inputs=inputs, outputs=out)
    return demo




if __name__ == "__main__":
    with gr.TabbedInterface(
            [text2image_gr()],
            ["文到图搜索"],
    ) as demo:
        demo.launch(
            enable_queue=True,
)