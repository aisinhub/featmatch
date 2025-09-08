from pathlib import Path
from PIL import Image
from torchvision import transforms as T

def read_img(file, shape, save_if_not_exist=True):
    file_ = Path(str(file).replace(file.parts[-4], f"{file.parts[-4]}-{shape}"))
    
    if file_.exists():
        tmp = Image.open(file_)
        x = tmp.copy()
        tmp.close()
    elif save_if_not_exist:
        file_.parent.mkdir(parents=True, exist_ok=True)
        x = Image.open(file).convert('RGB')
        resize = T.Compose([T.Resize(shape, Image.LANCZOS), T.CenterCrop(shape)])
        x = resize(x)
        x.save(file_)
    else:
        x = Image.open(file).convert('RGB')
        resize = T.Compose([T.Resize(shape, Image.LANCZOS), T.CenterCrop(shape)])
        x = resize(x)

    return x

def resize_single_image(file_path, output_size):
    """
    指定された単一の画像ファイルをリサイズし、新しいファイルとして保存する
    
    :param file_path: リサイズしたい画像ファイルのパス (str or Path)
    :param output_size: リサイズ後の画像のサイズ (int)
    """
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"Error: File not found at {file_path}")
        return

    # read_img関数を利用してリサイズと保存を行う
    print(f"Resizing {file_path} to {output_size}x{output_size}...")
    read_img(file_path, output_size)
    print("Done.")

# 使用例
if __name__ == '__main__':
    # ここにリサイズしたいファイルのパスとサイズを指定
    single_image_path = "/home/hirakawa/DS923_v2/hirakawa_ws_DS923_v2/dataset/svhn_format2/real/watermelon/real_335_000602.jpg"
    target_size = 256
    
    resize_single_image(single_image_path, target_size)