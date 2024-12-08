# 실험 목록

| | **실험 제목**                                         | **주요 내용**                                                                                  |
|----------|-------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| 1        | Auxiliary Classifier 적용 실험                         | 중간 계층의 Feature Map을 활용한 보조 분류기를 추가하여 성능 개선 여부를 확인                  |
| 2        | 최적의 Loss 찾기 실험                                  | BCE Loss, Dice Loss, Focal Loss 등 다양한 Loss를 비교하여 Semantic Segmentation에 적합한 Loss 탐색 |
| 3        | Pseudo Labeling                                       | 모델이 생성한 Pseudo Label을 학습 데이터에 포함하여 Test Set과 Train Set 간 일반화 성능 확인    |
| 4        | Pixel Shuffling, Unshuffling을 통한 interpolate 대안 실험 | Down Conv Layer와 Up Conv Layer를 추가하여 Resize 시 정보 손실 문제 해결을 시도               |
| 5        | 겹치는 손등 뼈 부분 정확도 개선하기 위한 1x1 Conv Layer 추가 | Output Mask 뒤에 1x1 Convolution Layer를 추가하여 특정 Class의 Threshold 조정을 통한 정확도 개선 |

<br>
<br>

## 1) Auxiliary Classifier 적용 실험

### **가설**

Auxiliary Classifier는 중간 계층의 Feature Map을 기반으로 추가적인 예측을 수행하도록 설계된 보조 분류기이다.  
선행 논문 조사에서 의료 영상 분석에 Auxiliary Classifier를 적용한 결과, 기존 방법들보다 더 높은 정확성을 달성했다.  
이에 현재 Hand Bone Dataset에서도 성능 향상을 기대하며 Auxiliary Classifier를 적용해보았다.

---

### **방법**

1. Classifier를 직접 확인 및 수정 (29개 클래스 개수에 맞게 설정)
2. Train 함수 수정
   - Main Loss와 Aux Loss에 가중치를 부여하여 Total Loss를 계산
   - Total Loss로 Backpropagation 수행

---

### **결론**

1. **FCN ResNet50**에서는 Auxiliary Classifier를 적용하여 성능이 **0.0005** 상승
2. **DeepLabv3 ResNet101**에서는 Auxiliary Classifier를 적용하지 않은 경우가 가장 높은 성능을 기록
3. Auxiliary Classifier의 가중치 Alpha를 0.4에서 0.7로 조정한 결과, 성능이 오히려 **0.0016** 감소  
   이를 통해 Auxiliary Classifier의 비율(Alpha)은 낮추는 것이 더 적합하다고 판단

#### **결과 테이블**

##### FCN ResNet50
| Model          | Aux Classifier | Val Score | Public Score | Private Score |
|----------------|----------------|-----------|--------------|---------------|
| FCN_ResNet50   | -              | 0.9444    | **0.9428**   | **0.9438**    |
| FCN_ResNet50   | alpha 0.4      | 0.9449    | 0.9424       | 0.9434        |

##### DeepLabv3 ResNet101
| Model                | Aux Classifier | Val Score | Public Score | Private Score |
|----------------------|----------------|-----------|--------------|---------------|
| DeepLabv3_ResNet101  | -              | 0.9472    | **0.9446**   | **0.9457**    |
| DeepLabv3_ResNet101  | alpha 0.4      | 0.9473    | 0.9431       | 0.9451        |
| DeepLabv3_ResNet101  | alpha 0.7      | 0.9457    | 0.9424       | 0.9439        |

---

### **회고**

1. Hand Bone Image Segmentation은 세밀한 경계와 복잡한 형태의 구조를 가지며, 클래스 간 경계가 매우 가까운 경우가 많다. Auxiliary Classifier는 주로 중간 계층에서 보다 일반적인 Feature를 학습하는 경향이 있는데, 현재는 네트워크 깊이가 상대적으로 얕아서 좋은 정보를 학습하지 못 했고, Segmentation Model 특성상 Decoder 부분에서 H,W를 점점 크게 만드는 연산을 하는데, 그 중간에서 분류를 돌리면 세밀한 부분에 대해서 큰 Resolution 보다 더 안 좋은 성능을 내서 성능이 안 나온 것 같다.
2. 위에는 Auxiliary Classifier를 단독으로 사용했을 시, 성능이 안 나온 경우에 대한 회고였고, 선행논문에서는 그럼에도 불구하고 기존 Loss와 Auxiliary Classifier를 이용한 Loss를 합해서 썼는데, 논문은 OOD에 관한 연구여서 현재 Task와 달라서 결과가 다르게 나왔을 수 있었을 거라 생각했다.
3. Task는 다르지만 위 논문에서 Lambda를 0.1로 준 것처럼, Auxiliary Classifier의 Loss에 0.1로 가중치를 적용해서 학습시켜보았으면 좋았을 거라는 아쉬움이 있다.

<br>

<br>

## 2) 최적의 Loss 찾기 실험

### **가설**

- Semantic Segmentation Task에서는 Pixel 단위로 정밀한 분류를 요구하기 때문에 단순한 Cross Entropy Loss가 모든 경우에 적합하지 않을 수 있다. 특히, 경계선 근처의 예측 정확도가 중요한 경우에는 이를 보완할 수 있는 특화된 Loss가 필요하다.
- Cross Entropy Loss는 Pixel별로 Loss를 계산하며, 모든 Pixel에 동일한 가중치를 부여한다.  
  따라서 배경 Pixel이 많은 과제에서는 배경 Pixel에 대한 Loss도 계산되어, Object가 적은 경우 모델이 0으로 출력하는 경향이 있다.  
  현재 Handbone X-ray Data는 배경이 많고 Object가 적으므로 이러한 문제가 발생할 가능성이 있다.

---

### **방법**

- 빠르게 실험할 수 있는 FCN_ResNet50 모델을 사용
- 다음 Loss 함수들에 대해 성능 비교:
  1. Focal Loss
  2. Dice Loss
  3. Tversky Loss
  4. Jaccard Loss
  5. BCE Loss + Dice Loss (Weight: 0.5)

---

### **결론**

- Dice Loss가 가장 좋은 성능을 기록

| Model         | Loss Function          | Val Score | Public Score | Private Score |
|---------------|------------------------|-----------|--------------|---------------|
| FCN_ResNet50  | BCE Loss               | 0.9444    | 0.9428       | 0.9438        |
| FCN_ResNet50  | Dice Loss              | 0.9497    | **0.9488**   | **0.9497**    |
| FCN_ResNet50  | BCE Loss(0.5) + Dice Loss(0.5)   | 0.9475    | 0.9433       | 0.9450        |
| FCN_ResNet50  | Tversky Loss           | 0.9488    | 0.9473       | 0.9481        |
| FCN_ResNet50  | Focal Loss             | 0.9472    | 0.9432       | 0.9444        |
| FCN_ResNet50  | Jaccard Loss           | 0.9494    | 0.9478       | 0.9486        |

---

### **회고**

- Focal Loss는 이미지 배경이 넓을 때 배경과 Object간의 Pixel 개수 불균형 해결에 적합했다고 생각한다.
- Dice Loss는 Class별로 생각하니, 배경에 대한 고려는 제외할 수 있고, 두 집합(예측된 Pixel과 실제 Pixel)의 교집합을 강조하기 때문에, 소수 Class의 학습 기여도를 높이고 균형 잡힌 학습을 유도가능했으며, 두 집합 간의 중첩 영역을 강조하기 때문에 경계선 근처의 작은 차이도 손실에 반영가능해서 성능이 잘 나왔다고 생각한다.
![image](https://github.com/user-attachments/assets/59103cea-306f-4293-a57a-6e5701f91895)   
- 다양한 Combined Loss를 실험했으면 좋았을 것이라 생각한다.

<br>

<br>

## 3) Pseudo Labeling

### **가설**

- Test Set과 Train Set은 동일한 X-ray 장비, 촬영 환경, 환자 그룹에서 수집되었거나 유사한 환경에서 생성된 것으로 가정한다.
- 따라서, 두 Dataset의 Handbone X-ray 시각적 특징과 Class 분포는 크게 다르지 않을 것이며, 학습된 모델이 Test Set에서도 일반화할 수 있을 것으로 기대한다.

---

### **방법**

1. **Output.csv 변환**: Output.csv의 RLE 형식을 Poly_seg 형식으로 변환
2. **Train 데이터 형식에 맞춤 저장**: 기존 Train 데이터 폴더 형식(Annos, Images)에 맞춰서 저장
3. **시각화**: Streamlit을 활용해 Pseudo Label 결과를 출력 및 확인

---

### **결론**

- Pseudo Labeling 적용 시 성능이 오히려 **0.0024** 감소했다.

| Model                      | Pseudo   | Public Score | Private Score |
|----------------------------|----------|--------------|---------------|
| BEiT_Large (BEiT_25k)      | -        | **0.9533**   | **0.9540**    |
| BEiT_Large (BEiT_25k)      | Pseudo   | 0.9509       | 0.9520        |

---

### **회고**

1. Model이 생성한 Pseudo Label이 부정확하거나, 잘못된 Label이 학습에 사용됐을 수 있다고 생각한다. 특히, X-ray Data는 경계가 모호하거나 미세한 구조를 구분해야 하므로, 작은 오류도 큰 영향을 미칠 수 있었을 거라 생각한다.
2. 29개 Class 중 일부 Class가 지나치게 많거나 적은 경우, Pseudo Label이 특정 Class에 편향될 수 있다. Model이 자주 등장하는 Class에만 초점을 맞추고 드물게 등장하는 Class의 Prediction 성능이 저하됐을거라 생각한다.

<br>

<br>

## 4) Pixel Shuffling, Unshuffling을 통한 interpolate 대안 실험

### **가설**

- 2048x2048 이미지를 1024x1024로 바로 Resize하면 성능 저하가 발생한다.
- **Encoder 앞단**에 Down Conv Layer를 추가하여 입력 이미지를 Down Sampling (Batch size, `Channel=3`, 2048, 2048) → (Batch size, `Channel=8`, 1024, 1024)하면, 입력이미지를 Down Sampling하고, 원본 해상도의 중요한 정보 유지할 것이라 가정했다.
- **Decoder 뒷단**에 Up Conv Layer를 추가하면 단순한 Up Sampling 방식이었던 Bilinear Interpolation보다 복잡한 세부 정보를 복원하는데 효과얻을 수 있으리라 가정했다.

---

### **방법**

1. **모델**:
   - Segformer_Mit_b1
   - DeepLabv3+ (ResNet101 Backbone)

2. **모델 수정**:
   - `mmseg > models > backbones > ${model}.py`에 새 Class를 생성하여 Down Conv Layer를 추가한다.
   - `mmseg > models > decode_heads > ${model_head}.py`에 새 Class를 생성하여 Up Conv Layer를 추가한다.
   - **DeepLabv3+** 의 경우, Backbone ResNet101의 맨 앞 Stem이 3채널을 받아 이미 Down Sampling을 진행하므로, Down Conv Layer를 추가할 필요가 없다고 판단하여 Decoder 뒷단에만 Up Conv Layer를 추가한다.
   - **Segformer** 의 경우, 추가된 Layer에 맞춰 첫 번째 PatchEmbed의 Projection을 수정했다.
   - **DeepLabv3+** 에서는 Sync Batch Norm을 Group Norm으로 변경했다.
   - Gradient Checkpoint를 적용한다.


---

### **결론**

- **Segformer_Mit_b1**:
  - Class별 Dice 계수가 5000 Iteration 이후에도 0으로 나타났다.
  - Up Conv Layer 또는 Down Conv Layer만 추가한 실험에서도 동일한 결과가 발생했다.

- **DeepLabv3+**:
  - 학습은 잘 되었으나, 기본 모델(Decoder에 Up Conv Layer 미적용)의 성능이 **0.9592**로 Up Conv Layer를 추가한 모델(0.9580)보다 더 높았다.
  - 3번째 실험 결과는 단순히 Iteration을 추가로 학습시켰기 때문에 성능이 올랐다고 판단했다.  

| Model                   | Val Score | Public Score | Private Score | Iter | Experiment                                              |
|-------------------------|-----------|--------------|---------------|------|--------------------------------------------------------|
| DeepLabv3+_ResNet101    | 0.9539    | **0.9592**   | **0.9647**    | 25k  | SyncBN → GN, Dice Loss                                 |
| DeepLabv3+_ResNet101    | 0.9444    | 0.9580       | 0.9626        | 25k  | Decoder 뒤에 Up Conv Layer, SyncBN → GN, Dice Loss     |
| DeepLabv3+_ResNet101    | 0.9676    | 0.9648       | 0.9675        | 31k  | Decoder 뒤에 Up Conv Layer, SyncBN → GN, Dice Loss<br>25k 이후 6k 추가 학습 |

---

### **회고**

1. **Segformer 학습 실패 원인 분석** :
   - **Mit Encoder**와 **SegFormer Decoder**는 원래 효과적으로 Feature를 활용하도록 설계되어 있어서
     - Encoder 앞의 Conv Layer가 중요한 Feature를 손실했을 가능성
     - Decoder 뒤의 Up Conv Layer가 Segmentation 결과를 왜곡했을 가능성
   - **PatchEmbed 수정** 과정에서 Weight 전달 문제가 발생했을 수 있다.
   - 경량화된 모델 특성상, 추가된 Conv Layer가 오히려 학습을 방해했을 가능성이 있다.

2. **DeepLabv3+ 학습 결과 회고** :
   - DeepLabv3+가 2048x2048 이미지가 들어갔을 때, 2048x2048의 Output을 출력하는 모델이었다면, Up Conv Layer를 다는게 오히려 방해가 되어 위와 같은 결과가 나왔다고 생각한다. 모델을 선택할 때, 어느정도 큰 모델 중에서 Input이 2048x2048일 때, 1024x1024로 Output이 나와서 그걸 Bilinear하는 모델을 골랐어야 했다고 생각한다. 그 확인 과정을 누락한 것 같다.

<br>

<br>

## 5) 겹치는 손등 뼈 부분 정확도 개선하기 위한 1 by 1 Conv Layer를 추가하는 실험

### **가설**

- Output Mask의 각 Channel은 해당 Class에 속할 Logits이다.
- **1x1 Convolution Layer**는 29개의 입력 Channel 값을 받아, 새로운 29개의 출력 Channel 값을 생성한다.
- Output Mask 이후에 1x1 Convolution Layer를 추가하면, 각 Channel의 가중치를 학습하여 특정 Class의 Threshold를 조정하고, 손등 뼈 부분에서의 겹침으로 인한 정확도 저하를 개선할 수 있을 것으로 가정했다.

---

### **방법**

- Decoder 뒤에 1x1 Convolution Layer를 추가하여 각 채널에 대해 Weight를 연산하고 Mask를 계산한다.

---

### **결론**

- 1x1 Convolution Layer를 추가한 결과
  - **25k Iter**에서 성능이 0.9618로, 기존 0.9592보다 **0.0026** 상승했다.
  - **37k Iter**에서 성능이 0.9653으로, 기존 0.9642보다 **0.0011** 상승했다.
  - **50k Iter**에서는 성능이 오히려 0.9641로 감소했다.

| Model                  | Val Score | Public Score | Private Score | Iter | Experiment                                   |
|------------------------|-----------|--------------|---------------|------|---------------------------------------------|
| DeepLabv3+_ResNet101   | 0.9539    | 0.9592       | 0.9647        | 25k  | SyncBN → GN, Dice Loss                      |
| DeepLabv3+_ResNet101   | 0.9445    | **0.9618**   | 0.9647        | 25k  | SyncBN → GN, Dice Loss, 1 by 1 Conv Layer   |

| Model                  | Val Score | Public Score | Private Score | Iter | Experiment                                   |
|------------------------|-----------|--------------|---------------|------|---------------------------------------------|
| DeepLabv3+_ResNet101   | 0.9647    | 0.9642       | 0.9674        | 37k  | SyncBN → GN, Dice Loss (25k 이후 12k 더 학습) |
| DeepLabv3+_ResNet101   | 0.9666    | **0.9653**   | **0.9677**    | 37k  | SyncBN → GN, Dice Loss, 1 by 1 Conv Layer (25k 이후 12k 더 학습) |

| Model                  | Val Score | Public Score | Private Score | Iter | Experiment                                   |
|------------------------|-----------|--------------|---------------|------|---------------------------------------------|
| DeepLabv3+_ResNet101   | 0.9665    | 0.9641       | 0.9664        | 50k  | SyncBN → GN, Dice Loss, 1 by 1 Conv Layer (25k 이후 25k 더 학습) |

---

### **회고**

- 1x1 Convolution Layer를 추가하는 것이 효과가 있는 것은 확인했으나, 팀 내에서 가장 좋은 모델이었던 Segformer에 실험해보면 더 좋은 성능이 나오지 않았을까 생각한다.
- Kernel 사이즈를 3x3으로 늘려가면서 실험하거나, Convolution Layer 대신에 Attention Layer를 추가하는 실험을 했으면 더 다양한 비교가 가능하지 않았을까 생각한다.
