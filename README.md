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
# Usage
**1.** To run the entire code, enter this command.
```
python main.py
```
<br>

**2.** If you want to save the CNN model weights after training, enter this command.

```
python main.py --save_model 1 --save_weights cnn_weights.hdf5
```  
<br>

**3.** To load the saved model weights and avoid the training time again, enter one of these commands.

```
python main.py --load_model 1 --save_weights cnn_weights.hdf5
```
```
python main.py -m 1
```
<br>


![HDR Cap4](https://user-images.githubusercontent.com/68259786/152900940-f20de78a-b2cd-415b-a088-3090b9d4d58e.JPG)

