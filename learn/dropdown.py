import gradio as gr

languages = ["spanish", "english"]
homeworks = {
    "spanish": ["hola", "bien", "gracias"],
    "english": ["hello", "good", "thank you"],
}


def rs_change(rs):
    return gr.update(choices=homeworks[rs], value=None)


with gr.Blocks() as app:

    rs = gr.Dropdown(choices=languages, value="english")
    rs_hw = gr.Dropdown(choices=homeworks["english"], interactive=True)

    rs.change(fn=rs_change, inputs=[rs], outputs=[rs_hw])

app.launch()
