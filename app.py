import gradio as gr
from text2image import text2image_gr

if __name__ == "__main__":
    app = gr.TabbedInterface(
        interface_list=[text2image_gr()],  # 子界面列表
        tab_names=["图文检索"],  # 标签名称
        title='clip图文检索项目'
    )

    app.launch(server_name="0.0.0.0", server_port=5001)