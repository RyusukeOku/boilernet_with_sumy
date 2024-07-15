#! /usr/bin/python3


import argparse
import pickle
import os
import pickle
import csv
import math
import numpy as np
import json


from .leaf_classifier import LeafClassifier
from .train import get_dataset, get_class_weights
from .misc import util
from .preprocess import get_leaves, parse, parse_doc, parse_doc_fromurl
import tensorflow as tf
from tensorflow.keras.models import load_model

tf.config.set_visible_devices([], 'GPU')#tensorflowがGPUを探しにいくのを防止

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-w','--working_dir', default='train', help='Where to save checkpoints and logs')
    ap.add_argument('-i','--predict_input_dir', help='Directory of files produced by the preprocessing script')
    ap.add_argument('-o','--predict_output_dir', help='Directory of files produced by the preprocessing script')
    ap.add_argument('-r','--predict_raw_dir', help='Directory of files produced by the preprocessing script')
    ap.add_argument('-p','--checkpoint_number', default=9, help='checkpoint path of model')
    ap.add_argument('-b','--balance', type=float, default=0, help='The dropout percentage')

    args = ap.parse_args()
    #  args = ap.parse_args(args=["../googletrends_japanese_data_5000","--working_dir","../googletrends_japanese_train_5000", "--test_dir", "../local_data_japan"])

    params_path = os.path.join(args.working_dir, 'params.csv')
    params = {}
    with open(params_path, 'r') as f:
        for i in csv.reader(f):
            params[i[0]] = i[1]

    DATA_DIR = params["DATA_DIR"]
    #文章をノードごとに分割
    file_path =args.predict_raw_dir
    doc_data, _, _ = parse_doc(util.get_filenames(file_path))

    info_file = os.path.join(DATA_DIR, 'info.pkl')
    with open(info_file, 'rb') as fp:
        info = pickle.load(fp)
        info['num_train_examples'] = len(doc_data)
        train_steps = math.ceil(info['num_train_examples'] / int(params["batch_size"]))

    predict_set_file = os.path.join(args.predict_input_dir, 'train.tfrecords')
    predict_dataset = get_dataset(predict_set_file, 1, repeat=False, shuffle=False)

    kwargs = {'input_size': info['num_words'] + info['num_tags'],
                'hidden_size': int(params['hidden_units']),
                'num_layers': int(params['num_layers']),
                'dropout': float(params['dropout']),
                'dense_size': int(params['dense_size'])}
    clf = LeafClassifier(**kwargs)

    checkpoint_path = args.working_dir + "/ckpt/model.{:03d}.h5".format(int(args.checkpoint_number))
    clf.model = load_model(checkpoint_path)

    _,y_pred = clf.eval_recall(predict_dataset, train_steps, desc="", balance=args.balance)

    filenames = util.get_filenames(file_path)
    raw_filenames = [i.split("/")[-1] for i in filenames]

    #予測結果を保存
    pred_index = [bool(i) for i in y_pred]
    delete_index = [not bool(i) for i in y_pred]
    counter = 0
    for i in range(len(raw_filenames)):
        numpy_data = np.array(doc_data[raw_filenames[i]])
        contents = numpy_data[pred_index[counter:counter+len(numpy_data)]]
        delete_contents = numpy_data[delete_index[counter:counter+len(numpy_data)]]
        counter += len(numpy_data)
        content_text = ""
        delete_text = ""
        for content in contents:
            content_text += str(content) + "\n"
        print("content:\n")
        print(content_text)
        for delete_content in delete_contents:
            delete_text += str(delete_content) + "\n"
        dir_path = args.predict_output_dir
        os.makedirs(dir_path, exist_ok=True)
        with open(os.path.join(dir_path, raw_filenames[i].replace("/", "_").replace(".", "_") + ".txt"), 'w') as file:
                file.write(content_text)
        with open(os.path.join(dir_path, raw_filenames[i].replace("/", "_").replace(".", "_") + "delete.txt"), 'w') as file:
                file.write(delete_text)

def predict_url(working_dir='train', predict_input_dir=None, urls=None, checkpoint_number=9, balance=0, docs=None):
    
    params_path = os.path.join(working_dir, 'params.csv')
    params = {}
    with open(params_path, 'r') as f:
        for i in csv.reader(f):
            params[i[0]] = i[1]

    DATA_DIR = params["DATA_DIR"]
    #文章をノードごとに分割
    #file_path =predict_raw_dir
    doc_data, _, _ = parse_doc_fromurl(urls=urls,docs=docs)

    info_file = os.path.join(DATA_DIR, 'info.pkl')
    with open(info_file, 'rb') as fp:
        info = pickle.load(fp)
        info['num_train_examples'] = len(doc_data)
        train_steps = math.ceil(info['num_train_examples'] / int(params["batch_size"]))

    predict_set_file = os.path.join(predict_input_dir, 'train.tfrecords')
    predict_dataset = get_dataset(predict_set_file, 1, repeat=False, shuffle=False)

    kwargs = {'input_size': info['num_words'] + info['num_tags'],
                'hidden_size': int(params['hidden_units']),
                'num_layers': int(params['num_layers']),
                'dropout': float(params['dropout']),
                'dense_size': int(params['dense_size'])}
    clf = LeafClassifier(**kwargs)

    checkpoint_path = working_dir + "/ckpt/model.{:03d}.h5".format(int(checkpoint_number))
    clf.model = load_model(checkpoint_path)

    _,y_pred = clf.eval_recall(predict_dataset, train_steps, desc="", balance=balance)


    #予測結果を保存
    pred_index = [bool(i) for i in y_pred]
    delete_index = [not bool(i) for i in y_pred]
    counter = 0
    count=1#urlの数
    all_content_text = ""  # すべてのURLに対するcontent_textを結合するための変数
    counter = 0
    count = 1  # urlの数
    for i in range(len(urls)):
        numpy_data = np.array(doc_data[str(i) + "page"])
        contents = numpy_data[pred_index[counter:counter + len(numpy_data)]]
        delete_contents = numpy_data[delete_index[counter:counter + len(numpy_data)]]
        counter += len(numpy_data)
        content_text = ""
        delete_text = ""
        for content in contents:
            content_text += str(content) + "\n"
        all_content_text += content_text  # すべてのcontent_textを結合
    
    return all_content_text  # すべてのcontent_textを返す


if __name__ == '__main__':
    main()
