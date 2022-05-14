# Handwritten Digit Recognition
This project is a project that recognizes numbers using deep learning models.<br>
Refer to 'Handwritten Digit Recognition using ML and DL(IJARCET-VOL-6-ISSUE-7-990-997)', 'Codetorial'.<br>
<br>
<br>

#### 혼자 진행한 프로젝트<br>
2019.06.01.~2019.06.03.<br>
<br>
직접 딥러닝 모델을 이용하여 숫자를 인식해보고 싶어<br>
혼자 간단한 프로젝트를 진행하였습니다.<br>
<br>
'Handwritten Digit Recognition using ML and DL(IJARCET-VOL-6-ISSUE-7-990-997)'와<br>
'Codetorial'의 코드를 참고하였습니다.<br>
<br>
<br>

# Usage (사용법) 
**1.** To run the entire code, enter this command. (전체 코드를 돌리기 위한 명령어)
```
python main.py
```
<br>

**2.** If you want to save the CNN model weights after training, enter this command. (훈련 후 CNN 모델의 가중치 파일을 저장하기 위한 명령어)
```
python main.py --save_model 1 --save_weights cnn_weights.hdf5
```  
<br>

**3.** To load the saved model weights and avoid the training time again, enter one of these commands. (CNN 모델의 가중치 파일을 불러와 훈련을 다시 하지 않게 하기 위한 명령어)
```
python main.py --load_model 1 --save_weights cnn_weights.hdf5
```
```
python main.py -m 1
```

<br>
<br>

# 프로젝트 실행결과
#### 초기 화면<br>
<img src="/images/image01.JPG" width="300"><br>
<br>
#### 숫자 그리기<br>
<img src="/images/image02.JPG" width="300"><br>
<br>
#### Predict 버튼을 누른 후에 학습하여 예측 결과를 도출<br>
<img src="/images/image03.JPG" width="600"><br>
<br>
