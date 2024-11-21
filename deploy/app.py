
__all__ = ['learn','classify_image','categories','image','label','examples','demo']

# import libraries
from fastai.vision.all import *
import gradio as gr
import pathlib

# load model
learn = load_learner('export.pkl')

# load categories
categories = ('Character','Scenery')

# define function
def classify_image(img):
    # input image
    # output prediction
    # uses prediction model to return a string of either pixel or scenary classification

    pred,idx,probs = learn.predict(img)
    return dict(zip(categories,map(float,probs)))

# define gradio interface
image = gr.Image(width=192, height=192)
label = gr.Label()
examples = ['char_example.jpg','sce_example.jpg']

# launch gradio interface
demo = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
demo.launch()