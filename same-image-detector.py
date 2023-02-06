import torch
import torch.nn as nn
from img2vec_pytorch import Img2Vec
from PIL import Image
import yaml
import os
from pathlib import Path

# 設定の読み込み
settings_file_path = Path(os.path.dirname(__file__)) / 'settings.yaml'
settings = yaml.safe_load(open(settings_file_path, encoding='utf-8'))

# Initialize Img2Vec with GPU
img2vec = Img2Vec(cuda=True)
cossim = nn.CosineSimilarity()
cossim_threshold: float = settings['cossim_threshold']

def is_image_path(path) -> bool:
    suffix = Path(path).suffix
    return suffix in ['.png', '.jpg'] # とりあえずpngとjpg

# 検索対象ディレクトリの列挙
def get_directory_paths(root_dir_path: Path) -> list[Path]:
    directory_paths = [root_dir_path]
    for dir in [p for p in root_dir_path.iterdir() if p.is_dir()]:
        directory_paths.extend(get_directory_paths(dir))
    return directory_paths
root_dir_path = Path(settings['root_dir_path'])
directory_paths = get_directory_paths(root_dir_path)

# 検索
print('')
for directory_path in directory_paths:
    files = os.listdir(directory_path)
    files = [file for file in files if is_image_path(file)]
    paths = [directory_path / file for file in files]
    path_and_vec: list[tuple[Path, torch.Tensor]] = []
    path_to_imagesize: dict[Path, tuple[int, int]] = {}
    found = False
    for path in paths:
        image = Image.open(path).convert('RGB')
        path_to_imagesize[path] = image.size
        vec = img2vec.get_vec(image, tensor=True)
        path_and_vec.append((path, vec))
    n = len(path_and_vec)
    for i in range(n):
        for j in range(i + 1, n):
            path1, vec1 = path_and_vec[i]
            path2, vec2 = path_and_vec[j]
            sim = float(cossim(vec1, vec2))
            if sim > cossim_threshold:
                if not found:
                    print(f'{path1.parent}:')
                    found = True
                print(str(path1.name), str(path2.name), round(sim, 3), end='')
                if path_to_imagesize[path1] != path_to_imagesize[path2]:
                    print('', path_to_imagesize[path1], path_to_imagesize[path2])
                else:
                    print('')
    if found:
        print('')
