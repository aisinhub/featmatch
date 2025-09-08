import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

def visualize_tsne_from_csv(csv_file_path, output_image_path='tsne_visualization.jpeg'):
    """
    CSVファイルから128次元ベクトルを読み込み、t-SNEで2次元に可視化し、
    JPEG画像として保存します。

    Args:
        csv_file_path (str): 10x128次元ベクトルが格納されたCSVファイルのパス。
        output_image_path (str): 保存するJPEG画像のファイルパス。
    """
    # 1. CSVファイルの読み込み
    try:
        # ヘッダーがない場合を想定してheader=None
        data = pd.read_csv(csv_file_path, header=None)
        features = data.values # pandas DataFrameからNumPy配列に変換
        print(f"Successfully loaded data from {csv_file_path}")
        print(f"Shape of loaded features: {features.shape}")
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_file_path}")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # データの形状チェック (10行128列を想定)
    if features.shape != (10, 128):
        print(f"Warning: Expected features shape (10, 128) but got {features.shape}.")
        print("t-SNE will still proceed, but check your input data if this is unexpected.")

    # 2. t-SNEの適用
    print("Applying t-SNE for dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=5, learning_rate=200, n_iter=1000)
    # n_components=2: 2次元に削減
    # random_state: 結果の再現性を確保
    # perplexity: データの密度に関するパラメータ。データ点数が少ない（10点）場合は、
    #             データ点数-1より小さくする必要があります（ここでは5を設定）。
    # learning_rate: 学習率
    # n_iter: 繰り返し回数

    try:
        tsne_results = tsne.fit_transform(features)
        print("t-SNE completed.")
        print(f"Shape of t-SNE results: {tsne_results.shape}")
    except ValueError as e:
        print(f"Error during t-SNE computation: {e}")
        print("Check 'perplexity' parameter, it must be less than the number of samples.")
        return
    except Exception as e:
        print(f"An unexpected error occurred during t-SNE: {e}")
        return

    # 3. 結果の可視化
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1])

    # 各点にラベル（例: Point 0, Point 1, ...）を付ける
    for i, txt in enumerate(range(features.shape[0])):
        plt.annotate(f'Point {txt}', (tsne_results[i, 0], tsne_results[i, 1]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.title('t-SNE Visualization of 128-dim Vectors')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True)
    plt.tight_layout()

    # 4. JPEG画像として保存
    try:
        plt.savefig(output_image_path, format='jpeg', dpi=300) # dpiで解像度を指定
        print(f"t-SNE visualization saved successfully to {output_image_path}")
    except Exception as e:
        print(f"Error saving image: {e}")
        return
    
    plt.close() # メモリ解放のためにプロットを閉じる

if __name__ == '__main__':
    # ダミーのCSVファイルを作成 (例として使用)
    dummy_csv_file = '/home/FeatMatch/fxg_output.csv'    
    # プログラムの実行
    visualize_tsne_from_csv(dummy_csv_file, '/home/FeatMatch/tsne_output.jpeg')
    
    # 実行後、ダミーファイルを削除 (オプション)
    # os.remove(dummy_csv_file)
    # print(f"Removed dummy CSV file: {dummy_csv_file}")