import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

csv = pd.read_csv('./output/best_score.csv') ## 후처리 적용할 csv파일 경로

def fill_inner_holes(mask):
    """
    외곽은 유지하면서 안쪽 빈 공간만 채우는 함수.
    """
    # 외곽을 반전하여 외부를 찾음
    h, w = mask.shape
    flood_fill_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)  # OpenCV 요구 사항으로 가장자리 추가
    inverted_mask = cv2.bitwise_not(mask)  # 0과 1을 반전
    # Flood Fill로 외부 채우기
    cv2.floodFill(inverted_mask, flood_fill_mask, (0, 0), 1)

    # 내부 빈 공간 채우기
    filled_mask = ((inverted_mask) > 1).astype(np.uint8)

    return filled_mask

def largest_connected_component(mask):
    """
    채워진 마스크에서 가장 큰 연결 성분만 남기기.
    """
    # 연결 성분 분석
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    # 가장 큰 연결 성분 선택 (배경 제외)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest_mask = (labels == largest_label).astype(np.uint8)
    return largest_mask

def rle_to_mask(rle, shape):
    """ RLE 데이터를 디코딩하여 2D 마스크 배열 생성 """
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    rle_pairs = [int(x) for x in rle.split()]
    for idx in range(0, len(rle_pairs), 2):
        start_pixel = rle_pairs[idx]
        run_length = rle_pairs[idx + 1]
        mask[start_pixel:start_pixel + run_length] = 1
    return mask.reshape(shape)

def mask_to_rle(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0]
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

shape = (2048,2048)
res = []

csv = csv.fillna('')

for i in csv.itertuples():
    if not i.rle:
        print(i.image_name, i._2, i.Index//29, 0)
        res.append({'image_name': i.image_name,
                    'class': i._2,
                    'rle': final_rle
                    })
        continue
    mask = rle_to_mask(i.rle, shape)
    filled_mask = fill_inner_holes(mask)
    final_mask = largest_connected_component(filled_mask)
    final_rle = mask_to_rle(final_mask)
    
    res.append({'image_name': i.image_name,
                'class': i._2,
                'rle': final_rle
                })
    if mask.sum() != final_mask.sum():
        print(i.image_name, i._2, i.Index//29, int(mask.sum())-int(final_mask.sum()))

pd.DataFrame(res).to_csv('postprocessed_output.csv',index=False)