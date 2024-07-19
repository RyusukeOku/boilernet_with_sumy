# Boilernet-with-Sumy

## 概要
Boilernet-with-Sumyは、指定されたURLから英語のWebページのテキストデータを取得し、その内容を5文に要約するツールです。このツールはAWS Lambda上で動作し、Dockerイメージとしてデプロイされます。
## 使用技術
このプロジェクトでは以下の技術とライブラリを使用しています：
- **boto3**: AWSサービスを操作するためのPythonライブラリ
- **beautifulsoup4**: HTMLやXMLファイルからデータをスクレイピングするためのライブラリ
- **html5lib**: HTMLを解析するためのライブラリ
- **nltk**: 自然言語処理のためのライブラリ
- **numpy**: 数値計算のためのライブラリ
- **scikit-learn**: 機械学習のためのライブラリ
- **tensorflow**: 機械学習と深層学習のためのライブラリ
- **tqdm**: プログレスバーを表示するためのライブラリ
- **scipy**: 科学技術計算のためのライブラリ
- **transformers**: トランスフォーマーモデルを使用するためのライブラリ
- **fugashi**: 日本語形態素解析のためのライブラリ
- **ipadic**: 日本語形態素解析辞書
- **scikit-learn**: 機械学習のためのライブラリ

## セットアップ方法
1. **リポジトリのクローン**:
    ```sh
    git clone https://github.com/yourusername/boilernet-with-sumy.git
    cd boilernet-with-sumy
    ```

2. **必要なライブラリのインストール**:
    ```sh
    pip install -r requirements.txt
    ```

3. **Dockerイメージのビルド**:
    ```sh
    docker build -t boilernet-with-sumy .
    ```

4. **AWS Lambdaへのデプロイ**:
    - AWS Lambdaコンソールにログインし、新しい関数を作成します。
    - 作成したDockerイメージをAWS ECRにプッシュし、Lambda関数にリンクします。

## 使い方
1. **Lambda関数のトリガー**:
    - AWS Lambdaコンソールで作成した関数を選択します。
    - テストイベントを設定し、以下のように入力します：
    ```json
    {
        "url": "https://example.com"
    }
    ```

2. **実行結果**:
    - Lambda関数を実行すると、指定されたURLのページが5文で要約されたテキストが出力されます。
