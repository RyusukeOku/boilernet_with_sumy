# coding: utf-8
import sys
import os
#import boto3
import json
import subprocess
from boilernet.net.preprocess import main_url
from boilernet.net.predict import predict_url
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

def summarize_text(text, sentences_count=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentences_count)
    summarized_text = ' '.join([str(sentence) for sentence in summary])
    return summarized_text


def handler(event, context):
    print("url:")
    print(event["url"])
    
    message="Nothing"
    # コマンドの構築
    command = [
        "python3",
        "boilernet/net/preprocess.py",
        "boilernet/html",
        "-w", "5000",
        "-t", "50",
        "--save", "/tmp/predict_data",
        "-td", "boilernet/preprocessed/googletrends_data",
        "-l", "English"
    ]
    
    command2 = [
    "python3",
    "boilernet/net/predict.py",
    "-w", "boilernet/working/googletrends_train",
    "-i", "/tmp/predict_data",
    "-o", "/tmp/output",
    "-r", "boilernet/html",
    "-p", "9"
    ]
    
    # コマンドを実行
    try:
        docs = []
        docs = main_url(urls=[event["url"]], num_words=5000, num_tags=50, save_dir="/tmp/predict_data", train_dir="boilernet/preprocessed/googletrends_data", language="English")
        all_content_text = predict_url(working_dir="boilernet/working/googletrends_train", predict_input_dir="/tmp/predict_data", urls=[event["url"]], checkpoint_number=9, docs=docs)

        summarized_text = summarize_text(all_content_text, sentences_count=5)

        #subprocess.run(command, check=True)
        #subprocess.run(command2, check=True)
        print("コマンドが正常に実行されました。")
        message = summarized_text
    except subprocess.CalledProcessError as e:
        print(f"エラーが発生しました: {e}")
        message=f"エラーが発生しました: {e}"
    return message

#handler({"url":"https://www.avforums.com/reviews/apple-iphone-15-pro-review.21249/"}, None)
