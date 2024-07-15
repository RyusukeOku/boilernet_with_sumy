# coding: utf-8
import sys
import os
import boto3
import json
import subprocess

def handler(event, context):
    print("url:\n")
    print(event["url"])
    
    message="Nothing"
    # コマンドの構築
    command = [
        "python3",
        "boilernet-master-enjp/net/preprocess.py",
        "boilernet-master-enjp/html",
        "-w", "5000",
        "-t", "50",
        "--save", "/tmp/predict_data",
        "-td", "boilernet-master-enjp/preprocessed/googletrends_data",
        "-l", "English"
    ]
    
    command2 = [
    "python3",
    "boilernet-master-enjp/net/predict.py",
    "-w", "boilernet-master-enjp/working/googletrends_train",
    "-i", "/tmp/predict_data",
    "-o", "/tmp/output",
    "-r", "boilernet-master-enjp/html",
    "-p", "9"
    ]
    
    # コマンドを実行
    try:
        subprocess.run(command, check=True)
        subprocess.run(command2, check=True)
        print("コマンドが正常に実行されました。")
        message="コマンドが正常に実行されました。"
    except subprocess.CalledProcessError as e:
        print(f"エラーが発生しました: {e}")
        message=f"エラーが発生しました: {e}"
    return message+'Hello from AWS Lambda using Python' + sys.version + '!'
