# Reduced MNIST 학습 파이프라인 서술형 2번

## A. 구현 결과 리포트 / 개선

### A-1. 데이터 split 방식 설명

MNIST 데이터셋을 사용했고, 전체 데이터 중 일부만 추려서 실험을 진행했다.  
구성은 다음과 같다.

- Train: 10,000
- Validation: 2,000
- Test: 2,000

Train과 Validation은 MNIST의 train split에서 나눴고,  
Test는 MNIST의 test split에서 따로 추출했다.

데이터 수가 많이 줄어드는 설정이라 클래스별 분포가 깨질 수 있다고 판단했다.  
그래서 stratified sampling을 사용해서, 각 숫자 클래스(0~9)가 고르게 포함되도록 했다.  
이를 통해 특정 숫자에 치우치지 않은 학습이 되도록 했다.

---

### A-2. Augmentation 설계 설명

Augmentation은 train 데이터에만 적용했다.  
Validation과 Test에는 augmentation을 적용하지 않고, 평가용 전처리만 사용했다.

Train 데이터에 적용한 augmentation은 다음과 같다.

- `RandomRotation(15)`  
  손글씨 숫자가 약간 기울어진 경우를 대비하기 위함이다.
- `RandomAffine(translate=(0.1, 0.1))`  
  숫자가 이미지 중앙에서 벗어나 있는 상황을 고려했다.

MNIST는 구조가 단순한 데이터셋이라,  
색상 변환이나 강한 변형은 오히려 도움이 되지 않을 수 있다고 판단했다.  
그래서 숫자 형태를 크게 해치지 않는 범위의 변형만 사용했다.

또한 ResNet pretrained 모델 입력에 맞추기 위해  
그레이스케일 이미지를 3채널로 변환하고, ImageNet 기준으로 정규화했다.

---

### A-3. 2단계 Transfer Learning (Freeze → Partial Unfreeze)

ImageNet으로 pretrained된 ResNet18을 사용했고,  
두 단계로 나누어 학습을 진행했다.

#### Phase 1

- Backbone은 모두 freeze하고, FC layer만 학습했다.
- 목적은 MNIST 분류에 맞게 classifier를 먼저 안정적으로 학습시키는 것이었다.
- 이 단계에서는 학습이 빠르게 수렴했고, validation 성능도 안정적으로 나왔다.

#### Phase 2

- `layer4`와 `fc`만 unfreeze해서 fine-tuning을 진행했다.
- 너무 많은 파라미터를 풀면 과적합 위험이 있다고 판단해, 상위 레이어만 선택했다.
- 이 단계에서 validation 성능이 조금 더 개선되었다.

---

### A-4. 학습 구성 방식 선정 이유

- Optimizer는 AdamW를 사용했다.  
  전이 학습 상황에서 과적합을 줄이는 데 도움이 된다고 판단했다.
- Loss는 CrossEntropyLoss를 사용했다.  
  다중 클래스 분류에서 가장 기본적이고 안정적인 선택이다.
- Learning rate는 backbone과 head를 다르게 설정했다.  
  backbone은 1e-5, head는 1e-3으로 설정해서  
  pretrained 가중치가 크게 변하지 않도록 했다.
- Batch size는 128로 설정했다.  
  CPU 환경에서 속도와 안정성을 고려한 값이다.

Scheduler는 사용하지 않았다.  
제한된 epoch 안에서 기본적인 수렴 특성을 확인하는 것을 우선으로 했다.

---

### A-5. 최종 성능 및 한계 분석

CPU 환경에서 전체 학습과 평가를 모두 수행했다.  
최종 결과는 다음과 같다.

- Best Validation Accuracy: **0.9785**
- Test Accuracy: **0.9825**
- Best Epoch: **7**

한계로는 다음과 같은 점이 있다.

- MNIST는 원래 28×28의 1채널 이미지인데,  
  3채널 224×224로 변환하는 방식은 연산 측면에서 비효율적이다.
- ResNet18은 MNIST에 비해 모델이 큰 편이라,  
  경량 모델을 쓰는 것보다 비효율적일 수 있다.
- CPU 환경이라 다양한 실험을 반복하기에는 시간이 오래 걸린다.

---

### A-6. 개선 아이디어

- Phase 2에서 learning rate scheduler를 적용해볼 수 있다.
- MobileNet 같은 경량 모델을 사용해 속도를 개선할 수 있다.
- Grad-CAM을 사용해 모델이 숫자의 어느 부분을 보고 판단하는지 확인해보고 싶다.

---

## B. Metric과 Loss

### B-1. CrossEntropyLoss와 Logits

CrossEntropyLoss는 내부적으로 softmax와 log 연산을 함께 처리한다.  
그래서 모델 출력이 확률이 아니라 logits 형태로 들어가는 것이 더 안정적이다.  
softmax를 따로 적용하면 수치적으로 불안정해질 수 있다.

---

### B-2. NLLLoss

NLLLoss는 log-probability를 입력으로 받는 loss다.  
보통 LogSoftmax를 먼저 적용한 뒤 사용한다.

CrossEntropyLoss는 LogSoftmax와 NLLLoss를 합쳐놓은 형태다.  
출력을 명시적으로 로그 확률로 다루고 싶을 때는 NLLLoss를 직접 쓰는 게 더 낫다.

---

### B-3. Softmax와 확률 해석

Softmax는 모델 출력을 확률처럼 해석할 수 있게 만든다.  
하지만 softmax 결과가 실제 정답일 확률과 완전히 같다고 보기는 어렵다.

모델이 틀린 예측에도 높은 확신을 가질 수 있기 때문에,  
실무에서는 calibration 문제가 중요해진다.

---

### B-4. Multi-class vs Multi-label

- Multi-class  
  - 하나의 정답만 존재
  - Softmax + CrossEntropyLoss
  - Accuracy 사용
- Multi-label  
  - 여러 정답 가능
  - Sigmoid + BCE Loss
  - F1-score, mAP 사용

---

### B-5. Label Smoothing

Label smoothing은 정답 라벨을 1로 두지 않고 조금 낮춰서 학습하는 방식이다.  
모델이 특정 정답에 너무 확신하지 않도록 도와준다.

일반화나 calibration 측면에서는 도움이 되지만 정답이 명확한 문제에서는 성능이 떨어질 수도 있다.

---

## C. 선택 질문

### C-2. Scheduler step 위치 차이

- 배치 단위 step은 학습률을 더 세밀하게 조절할 수 있다.
- 에폭 단위 step은 구현이 간단하다.
- validation 기준 step은 과적합을 줄이는 데 도움이 된다.

---

### C-5. Backbone을 freeze했는데 성능이 변하는 이유

1. BatchNorm 통계값은 학습 중에 계속 변할 수 있다.
2. Dropout은 학습 중에 랜덤성이 있다.
3. 데이터 순서나 augmentation 때문에 classifier 입력 분포가 달라질 수 있다.

---


## 실행 환경 정보

CPU 환경에서 전체 파이프라인을 끝까지 실행하여 결과를 도출하였다.

- OS: macOS 15.7.3
- Architecture: x86_64
- CPU: Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz

- Python version: 3.11.3
- PyTorch: 2.2.2
- NumPy: 1.26.4

- AMP: 설정은 되어 있으나 CUDA 미지원으로 자동 비활성화
- 실행 방식: 단일 프로세스, CPU 기반 end-to-end 학습
- CUDA: 사용 불가 (`torch.cuda.is_available() == False`)
- MPS: 사용 가능 (`torch.backends.mps.is_available() == True`)
- GPU device count: 0
