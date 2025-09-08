import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

def visualize_tsne_colored_by_class(csv_file_path, output_image_path='tsne_colored.jpeg'):
    """
    CSVファイルからクラスラベルと特徴ベクトルを読み込み、t-SNEで2次元に可視化し、
    クラスごとに色分けしてJPEG画像として保存します。

    Args:
        csv_file_path (str): 第1要素がクラス、第2要素以降が特徴ベクトルのCSVファイルのパス。
        output_image_path (str): 保存するJPEG画像のファイルパス。
    """
    # 1. CSVファイルの読み込み
    try:
        # ヘッダーがないことを想定してheader=None
        data = pd.read_csv(csv_file_path, header=None)
        
        # 第1列をクラスラベル、残りを特徴量として分離
        labels = data.iloc[:, 0].values.astype(int)
        features = data.iloc[:, 1:].values
        
        print(f"Successfully loaded data from {csv_file_path}")
        print(f"Shape of features: {features.shape}")
        print(f"Shape of labels: {labels.shape}")

    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_file_path}")
        return
    except Exception as e:
        print(f"Error reading or parsing CSV file: {e}")
        return

    # データのサンプル数が少ない場合のPerplexityチェック
    if features.shape[0] <= 1:
        print("Error: Not enough samples to perform t-SNE.")
        return
    perplexity = min(30, features.shape[0] - 1)
    if perplexity < 1:
        perplexity = 1

    # 2. t-SNEの適用
    print("Applying t-SNE for dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, learning_rate='auto', init='pca')
    try:
        tsne_results = tsne.fit_transform(features)
        print("t-SNE completed.")
        print(f"Shape of t-SNE results: {tsne_results.shape}")
    except ValueError as e:
        print(f"Error during t-SNE computation: {e}")
        print("Check 'perplexity' parameter. It must be less than the number of samples.")
        return
    except Exception as e:
        print(f"An unexpected error occurred during t-SNE: {e}")
        return

    # 3. 結果の可視化
    plt.figure(figsize=(12, 10))
    
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    colors = plt.cm.jet(np.linspace(0, 1, num_classes)) # クラス数に応じた色を生成
    
    for i, label in enumerate(unique_labels):
        # 各クラスに属するデータ点のみを抽出
        indices = labels == label
        
        # 散布図を描画
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1],
                    label=f'Class {label}', color=colors[i], alpha=0.8, s=50)

    plt.title('t-SNE Visualization with Class-based Coloring', fontsize=16)
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.legend(title='Classes', loc='best')
    plt.grid(True)
    plt.tight_layout()

    # 4. JPEG画像として保存
    try:
        plt.savefig(output_image_path, format='jpeg', dpi=300)
        print(f"t-SNE visualization saved successfully to {output_image_path}")
    except Exception as e:
        print(f"Error saving image: {e}")
        return
    
    plt.close()

if __name__ == '__main__':
    
    csv_file = '/home/FeatMatch/my_prog/cifar_10_eval.csv'    
    # プログラムの実行
    visualize_tsne_colored_by_class(csv_file, 'cifar_10_tsne.jpeg')
    
    