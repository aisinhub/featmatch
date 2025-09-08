import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def save_class_distance_histograms(filename, output_dir='my_prog'):
    """
    CSVファイルから各クラスの特徴ベクトルの平均からのユークリッド距離の分布をヒストグラムで可視化し、
    個別の画像ファイルとして保存します。
    
    Args:
        filename (str): クラスラベルと特徴ベクトルを含むCSVファイルのパス。
        output_dir (str): 画像ファイルを保存するディレクトリ名。
    """
    try:
        # CSVファイルを読み込む
        df = pd.read_csv(filename, header=None)
    except FileNotFoundError:
        print(f"エラー: ファイル '{filename}' が見つかりません。")
        return
        
    # 第1列をクラスラベル、残りを特徴ベクトルとして分離
    class_labels = df.iloc[:, 0].astype(int)
    features = df.iloc[:, 1:].values
    
    # 出力ディレクトリが存在しない場合は作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    unique_classes = np.unique(class_labels)
    
    for cls in unique_classes:
        # 現在のクラスに属する特徴ベクトルを抽出
        class_features = features[class_labels == cls]
        
        # クラス平均ベクトルを計算
        class_mean = np.mean(class_features, axis=0)
        
        # 各特徴ベクトルとクラス平均とのユークリッド距離を計算
        class_distances = np.linalg.norm(class_features - class_mean, axis=1)

        # プロットの作成
        plt.style.use('seaborn')
        plt.figure(figsize=(8, 6))
        
        sns.histplot(class_distances, bins=30, kde=True)
        
        # plt.title(f'クラス {cls} の特徴ベクトルと平均間のユークリッド距離の分布')
        plt.xlabel('distance')
        plt.ylabel('count')
        
        # ファイル名を設定して保存
        output_path = os.path.join(output_dir, f'class_{cls}_distance_histogram.png')
        plt.savefig(output_path)
        plt.close() # メモリを解放するために図を閉じる

        print(f"クラス {cls} のヒストグラムを {output_path} に保存しました。")


if __name__== '__main__':
    csv_filename = 'my_prog/cifar_10_eval.csv'
    save_class_distance_histograms(csv_filename)

