from rouge_score import rouge_scorer
from transformers import T5Tokenizer, T5ForConditionalGeneration
import matplotlib.pyplot as plt
from summarizer import Summarizer
from flask import Flask, redirect,url_for, request, render_template
app=Flask(__name__)

model_t5 = "t5-small"
tokenizerT5 = T5Tokenizer.from_pretrained(model_t5)
modelT5 = T5ForConditionalGeneration.from_pretrained(model_t5)

def summarize_Bert(text):
    bert_model = Summarizer()
    bert_summary = ''.join(bert_model(text, min_length=100))
    return bert_summary

def summarize_T5(text):
    inputs = tokenizerT5.encode(text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = modelT5.generate(inputs, max_length=150, num_beams=4, early_stopping=True)
    return tokenizerT5.decode(summary_ids[0], skip_special_tokens=True)
    
@app.route('/', methods=['GET', 'POST'])
def Summarize():
    if request.method == 'POST':
        inputText = request.form['text']
        summary = summarize_Bert(inputText)
        print(summary)
        return render_template('index.html', inputText = inputText, summary = summary)
    return render_template('index.html')

app.run(debug=False, port=80)