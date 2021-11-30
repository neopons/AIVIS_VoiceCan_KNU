# coding=utf-8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import sys
import kws_streaming.train.test as test
import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow as tf
import scipy as scipy
import scipy.io.wavfile as wav
import scipy.signal
from kws_streaming.models import models
from kws_streaming.models import utils
from kws_streaming.layers.modes import Modes
from argparse import Namespace
from collections import OrderedDict

def waveread_as_pcm16(filename):
  """Read in audio data from a wav file.  Return d, sr."""
  with tf.io.gfile.GFile(filename, 'rb') as file_handle:
    samplerate, wave_data = wav.read(file_handle)
  # Read in wav file.
  return wave_data, samplerate

def wavread_as_float(filename, target_sample_rate=16000):
  """Read in audio data from a wav file.  Return d, sr."""
  wave_data, samplerate = waveread_as_pcm16(filename)
  desired_length = int(
      round(float(len(wave_data)) / samplerate * target_sample_rate))
  wave_data = scipy.signal.resample(wave_data, desired_length)

  # Normalize short ints to floats in range [-1..1).
  data = np.array(wave_data, np.float32) / 32768.0
  return data, target_sample_rate

if __name__ == '__main__':
  wav_file = sys.argv[1]
  gender = sys.argv[2]
  age = sys.argv[3]
  model_subpath = sys.argv[4]
  predicted_word = sys.argv[5]
  if int(age) < 60:
    word_list = (sys.argv[6]).split(',')
  else:
    word_list = (sys.argv[7]).split(',')
  output_path = sys.argv[8]

  if len(sys.argv) != 9:
    print("mismatched arguments")
    sys.exit()
  
  if int(age) >= 60 and gender == 'male':
      model_path = model_subpath+'/crnn_stream_male'
  elif int(age) >= 60 and gender == 'female':
      model_path = model_subpath+'/crnn_stream_female'
  elif int(age) < 60 and gender == 'male':
      model_path = model_subpath+'/crnn_stream'
  elif int(age) < 60 and gender == 'female':
      model_path = model_subpath+'/crnn_stream'
  else:
      print("Cannot find model")
      sys.exit()

  with tf.compat.v1.gfile.Open(os.path.join(model_path, 'flags.txt'), 'r') as fd:
    flags_txt = fd.read()
  flags = eval(flags_txt)
        
  config = tf1.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf1.Session(config=config)
  tf1.keras.backend.set_session(sess)
  tf.keras.backend.set_learning_phase(0)  
  wav_data, samplerate = wavread_as_float(wav_file)
  if len(wav_data)<flags.desired_samples:
    padded_wav = np.pad(wav_data, (0, flags.desired_samples-len(wav_data)), 'constant')
  else:
    padded_wav = wav_data
  input_data = np.expand_dims(padded_wav, 0)
  
  model_non_stream_batch = models.MODELS[flags.model_name](flags)
  weights_name = 'best_weights'
  model_non_stream_batch.load_weights(os.path.join(model_path, weights_name))
  
  inference_batch_size = 1
  flags.batch_size = inference_batch_size

  model_stream = utils.to_streaming_inference(model_non_stream_batch, flags, Modes.STREAM_INTERNAL_STATE_INFERENCE)

  start = 0
  end = flags.window_stride_samples
  predicted_labels = int(word_list.index(predicted_word))+2
  first_step = True
  num=0
  best_num = 20
  threshold = 0.7
  max_percent = 0.0
  predicted_output=np.array([])
  
  while end <= input_data.shape[1]:
    stream_update = input_data[:, start:end]

    # get new frame from stream of data
    stream_output_prediction = model_stream.predict(stream_update)
    max_val = np.max(stream_output_prediction[0])
    exp_output = np.exp(stream_output_prediction[0]-max_val)
    sum_exp_output = np.sum(exp_output)
    output_data= exp_output/(sum_exp_output*1.0)
    
    if int(np.argmax(output_data)) == predicted_labels:
    #if output_data[predicted_labels] >= threshold:
      predicted_output = np.append(predicted_output, [output_data[predicted_labels]])
      num = num +1
      
    #if int(np.argmax(output_data)) == predicted_labels and output_data[predicted_labels] >= max_percent:
    #  max_percent = output_data[predicted_labels]
    #  max_output_data = output_data

    # update indexes of streamed updates
    start = end
    end = start + flags.window_stride_samples

  if num == 0:
    pred2 = 0
    pred3 = 0
  else:
    if num < best_num:
        best_num=num
    predicted_output_max = np.sort(predicted_output)[-best_num:]
    output=predicted_output_max.sum()/best_num*1.0
    pred3=output
    #print((output*100).astype(np.int32))
  #print((max_output_data*100).astype(np.int32))
  file_data = OrderedDict()
  file_data["filename"] = wav_file.split('/')[-1]
  file_data["keyword"] = predicted_word
  file_data["accuracy"] = str(pred3)
  with open(output_path+'/output.json', 'w', encoding="utf-8") as f:
    json.dump(file_data, f, ensure_ascii=False, indent="\t")
    print ('Done')

