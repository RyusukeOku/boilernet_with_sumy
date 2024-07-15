#! /usr/bin/python3


import os
import argparse
import pickle
import json
from collections import defaultdict

import nltk
import numpy as np
import tensorflow as tf
from bs4 import BeautifulSoup, Comment, NavigableString
from tqdm import tqdm

from .misc import util
#berttokenizerを追加
from transformers import BertJapaneseTokenizer

import requests

def get_leaves(node, tag_list=[], label=0):
    """Return all leaves (NavigableStrings) in a BS4 tree."""
    tag_list_new = tag_list + [node.name]
    if node.has_attr('__boilernet_label'):
        label = int(node['__boilernet_label'])

    result = []
    for c in node.children:
        if isinstance(c, NavigableString):
            # might be just whitespace
            if c.string is not None and c.string.strip():
                result.append((c, tag_list_new, label))
        elif c.name not in util.TAGS_TO_IGNORE:
            result.extend(get_leaves(c, tag_list_new, label))
    return result


def get_leaf_representation(node, tag_list, label, tokenize = None):
    """Return dicts of words and HTML tags that representat a leaf."""
    tags_dict = defaultdict(int)
    for tag in tag_list:
        tags_dict[tag] += 1
    words_dict = defaultdict(int)
    if tokenize is None:
        #print("before nltk.word_tokenize")
        tokenize = nltk.word_tokenize
        #print("after nltk.word_tokenize")
        words = tokenize(node.string)
        #print("aftertokenize")
    else:
        #print("here?")
        words = tokenize.convert_ids_to_tokens(tokenize.encode(node.string)[1:-1])
    for word in words:#nltk.word_tokenize(node.string):
        #print("word")
        words_dict[word.lower()] += 1
    return dict(words_dict), dict(tags_dict), label


def process(doc, tags, words, tokenize = None):
    """
    Process "doc", updating the tag and word counts.
    Return the document representation, the HTML tags and the words.
    """
    result = []
    for leaf, tag_list, is_content in get_leaves(doc.find_all('html')[0]):
        leaf_representation = get_leaf_representation(leaf, tag_list, is_content, tokenize =tokenize)
        result.append(leaf_representation)
        words_dict, tags_dict, _ = leaf_representation
        for word, count in words_dict.items():
            words[word] += count
        for tag, count in tags_dict.items():
            tags[tag] += count
    return result


def process_doc(doc, tags, words):
    """
    Process "doc", updating the tag and word counts.
    Return the document representation, the HTML tags and the words.
    """
    result = []
    for leaf, tag_list, is_content in get_leaves(doc.find_all('html')[0]):
        result.append(leaf)
    return result

def process_testdata(doc, tags, words, tokenize = None):
    """
    Process "doc", updating the tag and word counts.
    Return the document representation, the HTML tags and the words.
    """
    result = []
    for leaf, tag_list, is_content in get_leaves(doc.find_all('html')[0]):
        #print("before get_leaf_representation")
        leaf_representation = get_leaf_representation(leaf, tag_list, is_content, tokenize = tokenize)
        #print("finish get_leaf_representation")
        result.append(leaf_representation)
    return result

def parse_doc(filenames):
    """
    Read and parse all HTML files.
    Return the parsed documents and a set of all words and HTML tags.
    """
    result = {}
    tags = defaultdict(int)
    words = defaultdict(int)

    for f in tqdm(filenames):
        try:
            with open(f, 'rb') as hfile:
                doc = BeautifulSoup(hfile, features='html5lib')
            basename = os.path.basename(f)
            result[basename] = process_doc(doc, tags, words)
        except:
            tqdm.write('error processing {}'.format(f))
    return result, tags, words

def parse_doc_fromurl(urls=None,docs=None):
    """
    Read and parse all HTML files.
    Return the parsed documents and a set of all words and HTML tags.
    """
    result = {}
    tags = defaultdict(int)
    words = defaultdict(int)
    count=0
    for url in tqdm(urls):
        try:
            #headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2.1 Safari/605.1.15"}
            #response = requests.get(url, headers=headers)
            #response.raise_for_status()  # HTTPエラーが発生した場合に例外を発生させる
            #doc = BeautifulSoup(response.content, features='html5lib')
            doc = docs[count]
            basename = str(count)+"page"
            count+=1
            result[basename] = process_doc(doc, tags, words)
        except requests.RequestException as e:
            print(e)
            print("requests error")
            tqdm.write('error processing {}'.format(url))
        except:
            print("parse_testdata error")
            tqdm.write('error processing {}'.format(url))
    return result, tags, words


def parse(filenames, language = "English"):
    """
    Read and parse all HTML files.
    Return the parsed documents and a set of all words and HTML tags.
    """
    result = {}
    tags = defaultdict(int)
    words = defaultdict(int)
    tokenize = None
    if language == "Japanese":
        tokenize = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
        

    for f in tqdm(filenames):
        #try:
        with open(f, 'rb') as hfile:
            doc = BeautifulSoup(hfile, features='html5lib')
        basename = os.path.basename(f)
        result[basename] = process(doc, tags, words, tokenize = tokenize)
        # except:
        #     tqdm.write('error processing {}'.format(f))
    return result, tags, words

def parse_testdata(filenames, tags, words, language = "English"):
    """
    Read and parse all HTML files.
    Return the parsed documents and a set of all words and HTML tags.
    """
    result = {}
    tokenize = None
    if language == "Japanese":
        tokenize = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

    for f in tqdm(filenames):
        try:
            with open(f, 'rb') as hfile:
                doc = BeautifulSoup(hfile, features='html5lib')
            basename = os.path.basename(f)
            result[basename] = process_testdata(doc, tags, words, tokenize = tokenize)
        except:
            print("parse_testdata error")
            tqdm.write('error processing {}'.format(f))
    return result, tags, words

def parse_testdata_fromurl(urls, tags, words, language = "English"):
    """
    Read and parse all HTML files.
    Return the parsed documents and a set of all words and HTML tags.
    """
    result = {}
    tokenize = None
    if language == "Japanese":
        tokenize = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    count=0
    docs=[]
    for url in tqdm(urls):
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2.1 Safari/605.1.15"}
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # HTTPエラーが発生した場合に例外を発生させる
            doc = BeautifulSoup(response.content, features='html5lib')
            docs.append(doc)
            basename = str(count)+"page"
            count+=1
            result[basename] = process_testdata(doc, tags, words, tokenize = tokenize)
        except requests.RequestException as e:
            print(e)
            print("requests error")
            tqdm.write('error processing {}'.format(url))
        except:
            print("parse_testdata error")
            tqdm.write('error processing {}'.format(url))
    return result, tags, words, docs

def get_feature_vector(words_dict, tags_dict, word_map, tag_map):
    """Return a feature vector for an item to be classified."""
    vocab_vec = np.zeros(len(word_map), dtype='int32')
    for word, num in words_dict.items():
        # if the item is not in the map, use 0 (OOV word)
        vocab_vec[word_map.get(word, 0)] = num

    tags_vec = np.zeros(len(tag_map), dtype='int32')
    for tag, num in tags_dict.items():
        # if the tag is not in the map, use 0 (OOV tag)
        tags_vec[tag_map.get(tag, 0)] = num

    return np.concatenate([vocab_vec, tags_vec])


def get_vocabulary(d, num=None):
    """Return an integer map of the top-k vocabulary items and add <UNK>."""
    l = sorted(d.keys(), key=d.get, reverse=True)
    if num is not None:
        l = l[:num]
    int_map = util.get_int_map(l, offset=1)
    int_map['<UNK>'] = 0
    return int_map


def get_doc_inputs(docs, word_map, tag_map):
    """Transform "docs" into the input format accepted by the classifier."""

    def _int64_feature(l):
        """Return an int64_list."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=l))

    for doc in docs:
        doc_features = []
        doc_labels = []
        for words_dict, tags_dict, label in doc:
            feature_vector = get_feature_vector(words_dict, tags_dict, word_map, tag_map)
            doc_features.append(_int64_feature(feature_vector))
            doc_labels.append(_int64_feature([label]))
        doc_feature_list = tf.train.FeatureList(feature=doc_features)
        doc_label_list = tf.train.FeatureList(feature=doc_labels)
        yield doc_feature_list, doc_label_list


def write_tfrecords(filename, dataset, word_map, tag_map):
    """Write the dataset to a .tfrecords file."""
    with tf.io.TFRecordWriter(filename) as writer:
        for doc_feature_list, doc_label_list in get_doc_inputs(dataset, word_map, tag_map):
            f = {'doc_feature_list': doc_feature_list, 'doc_label_list': doc_label_list}
            feature_lists = tf.train.FeatureLists(feature_list=f)
            example = tf.train.SequenceExample(feature_lists=feature_lists)
            writer.write(example.SerializeToString())


def save(save_path, word_map, tag_map, train_set, dev_set=None, test_set=None):
    """Save the data."""
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, 'words.json'), 'w', encoding='utf-8') as fp:
        json.dump(word_map, fp,  ensure_ascii=False)

    with open(os.path.join(save_path, 'tags.json'), 'w', encoding='utf-8') as fp:
        json.dump(tag_map, fp,  ensure_ascii=False)

    info = {}
    info['num_words'] = len(word_map)
    info['num_tags'] = len(tag_map)

    train_file = os.path.join(save_path, 'train.tfrecords')
    print('writing {}...'.format(train_file))
    write_tfrecords(train_file, train_set, word_map, tag_map)
    info['num_train_examples'] = len(train_set)

    if dev_set is not None:
        dev_file = os.path.join(save_path, 'dev.tfrecords')
        print('writing {}...'.format(dev_file))
        write_tfrecords(dev_file, dev_set, word_map, tag_map)
        info['num_dev_examples'] = len(dev_set)

    if test_set is not None:
        test_file = os.path.join(save_path, 'test.tfrecords')
        print('writing {}...'.format(test_file))
        write_tfrecords(test_file, test_set, word_map, tag_map)
        info['num_test_examples'] = len(test_set)

    info_file = os.path.join(save_path, 'info.pkl')
    with open(info_file, 'wb') as fp:
        pickle.dump(info, fp)


def read_file(f_name):
    with open(f_name, encoding='utf-8') as fp:
        for line in fp:
            yield line.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('DIRS', nargs='+', help='A list of directories containing the HTML files')
    ap.add_argument('-s', '--split_dir', help='Directory that contains train-/dev-/testset split')
    ap.add_argument('-w', '--num_words', type=int, help='Only use the top-k words')
    ap.add_argument('-t', '--num_tags', type=int, help='Only use the top-l HTML tags')
    ap.add_argument('--save', default='result', help='Where to save the results')
    ap.add_argument('-td', '--train_dir', help='A list of directories containing the HTML files')
    ap.add_argument('-l', '--language', help='A list of directories containing the HTML files')
    args = ap.parse_args()

    filenames = []
    for d in args.DIRS:
        filenames.extend(util.get_filenames(d))
    if args.train_dir:
        with open(os.path.join(args.train_dir, 'words.json'), 'r', encoding='utf-8') as fp:
            words = json.load(fp)
        with open(os.path.join(args.train_dir, 'tags.json'), 'r', encoding='utf-8') as fp:
            tags = json.load(fp)
        data, tags, words = parse_testdata(filenames,tags,words,language=args.language)
    else:
        data, tags, words = parse(filenames, language=args.language)
        tags = get_vocabulary(tags, args.num_tags)
        words = get_vocabulary(words, args.num_words)

    if args.split_dir:
        train_set_file = os.path.join(args.split_dir, 'train_set.txt')
        dev_set_file = os.path.join(args.split_dir, 'dev_set.txt')
        test_set_file = os.path.join(args.split_dir, 'test_set.txt')
        train_set = [data[basename] for basename in read_file(train_set_file)]
        dev_set = [data[basename] for basename in read_file(dev_set_file)]
        test_set = [data[basename] for basename in read_file(test_set_file)]
    else:
        train_set = data.values()
        dev_set, test_set = None, None

    save(args.save, words, tags, train_set, dev_set, test_set)

def main_url(urls=None, split_dir=None, num_words=None, num_tags=None, save_dir='result', train_dir=None, language=None):

    # 現在のファイル（スクリプトまたはモジュール）の絶対パスを取得
    current_file_path = os.path.abspath(__file__)
    # 現在のファイルのディレクトリを取得
    current_directory = os.path.dirname(current_file_path)
    # 相対パスを絶対パスに変換
    nltk_data_path = os.path.join(current_directory, "../nltk_data")
    # 絶対パスがNLTKのデータパスにまだ追加されていない場合のみ追加
    if nltk_data_path not in nltk.data.path:
        nltk.data.path.append(nltk_data_path)
    
    docs=[]
    if train_dir:
        with open(os.path.join(train_dir, 'words.json'), 'r', encoding='utf-8') as fp:
            words = json.load(fp)
        with open(os.path.join(train_dir, 'tags.json'), 'r', encoding='utf-8') as fp:
            tags = json.load(fp)
        data, tags, words,docs = parse_testdata_fromurl(urls, tags, words, language=language)
    else:
        print("for train use is unsupoorted in this function")

    if split_dir:
        train_set_file = os.path.join(split_dir, 'train_set.txt')
        dev_set_file = os.path.join(split_dir, 'dev_set.txt')
        test_set_file = os.path.join(split_dir, 'test_set.txt')
        train_set = [data[basename] for basename in read_file(train_set_file)]
        dev_set = [data[basename] for basename in read_file(dev_set_file)]
        test_set = [data[basename] for basename in read_file(test_set_file)]
    else:
        train_set = list(data.values())
        dev_set, test_set = None, None

    save(save_dir, words, tags, train_set, dev_set, test_set)
    return docs


if __name__ == '__main__':
    #nltk.download('punkt')
    nltk.data.path.append("boilernet-master-enjp/nltk_data")#プログラムはlambda_function.pyがあるディレクトリから実行されること.
    print("Current Directory:", os.getcwd())
    print(nltk.data.path)
    main()
