import argparse

from omegaconf import OmegaConf
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(
        description='Ensemble inference tool')
    parser.add_argument('--config', help='inference config file path')
    parser.add_argument(
        '--method',
        choices=['majority_vote', 'average', 'intersection'],
        default='majority_vote',
        help='select ensemble method [majority_vote, average, intersection]'
    )
    args = parser.parse_args()
    return args

# mask map으로 나오는 인퍼런스 결과를 RLE로 인코딩 합니다.
def encode_mask_to_rle(mask):
    '''
    mask: numpy array binary mask
    1 - mask
    0 - background
    Returns encoded run length
    '''
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# RLE로 인코딩된 결과를 mask map으로 복원합니다.
def decode_rle_to_mask(rle, height=2048, width=2048):
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)

    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(height, width)

def majority_vote_ensemble(masks):
    """
    다수결 앙상블 함수.
    여러 바이너리 마스크를 입력받아 앙상블된 결과를 반환.
    """
    # 모든 마스크를 numpy 배열로 변환
    masks = np.stack(masks, axis=0)  # (n_models, height, width)

    # 다수결 기준: 각 픽셀에서 1의 비율이 0.5 이상이면 1로 설정
    return (np.mean(masks, axis=0) >= 0.5).astype(np.uint8)

def average_ensemble(masks, threshold=0.5):
    """
    평균값 기반 앙상블 함수.
    threshold: 평균 값이 해당 임계치 이상이면 1로 설정.
    """
    masks = np.stack(masks, axis=0)
    return (np.mean(masks, axis=0) >= threshold).astype(np.uint8)

def intersection_ensemble(masks):
    """
    교집합 방식의 앙상블 함수.
    """
    masks = np.stack(masks, axis=0)
    return np.all(masks, axis=0).astype(np.uint8)

def ensemble_masks_from_csv(csv_paths, height=2048, width=2048, method="majority_vote"):
    ensemble_data = {}
    
    # CSV 파일별로 처리
    for csv_path in tqdm(csv_paths, desc="Processing CSV files"):
        df = pd.read_csv(csv_path)
        
        for _, row in df.iterrows():
            image_name = row['image_name']
            class_id = row['class']
            rle = row['rle']

            mask = decode_rle_to_mask(rle, height=height, width=width)

            if image_name not in ensemble_data:
                ensemble_data[image_name] = {}
            if class_id not in ensemble_data[image_name]:
                ensemble_data[image_name][class_id] = []
            
            ensemble_data[image_name][class_id].append(mask)

    final_masks = {}
    for image_name, classes in tqdm(ensemble_data.items(), desc="Ensembling masks"):
        final_masks[image_name] = {}
        for class_id, masks in classes.items():
            if method == "majority_vote":
                final_masks[image_name][class_id] = majority_vote_ensemble(masks)
            elif method == "average":
                final_masks[image_name][class_id] = average_ensemble(masks)
            elif method == "intersection":
                final_masks[image_name][class_id] = intersection_ensemble(masks)
            else:
                raise ValueError(f"Unknown method: {method}")
    
    return final_masks

def main(args):
    cfg = OmegaConf.load(args.config)

    csv_paths = list(cfg.csv_paths)
    save_filename = str(cfg.save_filename)

    method = args.method

    final_masks = ensemble_masks_from_csv(csv_paths=csv_paths, method=method)
    
    rle_results = {}
    for image_name, classes in tqdm(final_masks.items(), desc="Encoding masks to RLE"):
        rle_results[image_name] = {}
        for class_id, mask in classes.items():
            rle_results[image_name][class_id] = encode_mask_to_rle(mask)

    rle_data = []
    for image_name, classes in tqdm(rle_results.items(), desc="Preparing final data"):
        for class_id, rle in classes.items():
            rle_data.append({
                "image_name": image_name,
                "class": class_id,
                "rle": rle
            })

    save_dir = "./ensemble_results"
    os.makedirs(save_dir, exist_ok=True)

    df = pd.DataFrame(rle_data)
    df.to_csv(f"{save_dir}/{save_filename}_{method}.csv", index=False)

    print("Completed")

    

if __name__ == "__main__":
    args = parse_args()
    main(args)



