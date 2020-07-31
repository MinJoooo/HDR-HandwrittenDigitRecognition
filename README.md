# HDR (Handwritten Digit Recognition)
Refer to 'Handwritten Digit Recognition using ML and DL(IJARCET-VOL-6-ISSUE-7-990-997)', 'Codetorial'

1. To run the entire code, enter this command.
  python main.py


2. If you want to save the CNN model weights after training, enter this command.
  > python main.py --save_model 1 --save_weights cnn_weights.hdf5


3. To load the saved model weights and avoid the training time again, enter one of these commands.
  python main.py --load_model 1 --save_weights cnn_weights.hdf5
  python main.py -m 1
