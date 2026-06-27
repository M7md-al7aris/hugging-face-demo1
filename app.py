from transformers import pipeline
import gradio as gr


model = pipeline("sentiment-analysis")


def predict(prompt):
    result = model(prompt)[0]
    output = f"Label: {result['label']} (Confidence: {result['score']:.4f})"
    return output


with gr.Blocks() as demo:
    textbox = gr.Textbox(placeholder="Enter a sentence to analize", lines=4)
    gr.Interface(fn=predict, inputs=textbox, outputs="text")

demo.launch()
