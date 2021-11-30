# SMARTEAR_KWS
we implement Convolutional Recurrent Neural Network (CRNN) for Keyword Spotting for VoiceCan.



<img width="746" alt="Model Architecture of CRNN" src="https://user-images.githubusercontent.com/18507848/138975595-bc17433a-5e62-477e-bcef-d10a184520f5.png">

## Getting Started
1. install requirements
2. run

### Prerequirites
__voice included wav_files__
__NN trainable server__
### Dataset : Google speech commands v0.02
- dataset link : http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
- also use noisy dataset (add wham noise)
- noisy-dataset link : https://drive.google.com/file/d/1loSDnPbCH0Do_lyTh14W_UXsuzACFxRR/view?usp=sharing
- refer to noisy-dataset-maker
### Dataset : Korean speech commands v0.03
- dataset link : https://drive.google.com/file/d/15ftx53o6scNbGk8eASGb7Kxu0x3d86q-/view?usp=sharing
- This Dataset based on 명령어 음성(일반남녀) 데이터 in AI-hub (명령어 음성(노인남녀) 데이터 명령어 음성(소아,유아)데이터 추가 예정)
- 100000~ speechs include 35 commands


### Installation
__check requirement.txt each directory__
## Usage

### Training
- train the various kws models including cnn, rnn, transformer.
- convert kws model to tflite model
