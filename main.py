import pygame
import pygame.midi
import random, math
import numpy as np
import cv2
import pyaudio
import midi
import wave
import time
from audiolazy import freq2midi

from leap import *



#User constants
device = "cpu"
dir_name = './music/History/'
sub_dir_name = 'e200/'
sample_rate = 48000
note_dt = 2000        #Num Samples
note_duration = 20000 #Num Samples
note_decay = 5.0 / sample_rate
num_params = 120
num_measures = 16
num_sigmas = 5.0
note_thresh = 32
use_pca = True
is_ae = True

background_color = (210, 210, 210)
edge_color = (60, 60, 60)
slider_colors = [(90, 20, 20), (90, 90, 20), (20, 90, 20), (20, 90, 90), (20, 20, 90), (90, 20, 90)]

note_w = 96
note_h = 96
note_pad = 2

notes_rows = num_measures / 8
notes_cols = 8

slider_num = min(40, num_params)
slider_h = 200
slider_pad = 5
tick_pad = 4

control_w = 210
control_h = 30
control_pad = 5
control_num = 3
control_colors = [(255,0,0), (0,255,0), (0,0,255)]
control_inits = [0.8, 0.5, 0.5]

#Derived constants
notes_w = notes_cols * (note_w + note_pad*2)
notes_h = notes_rows * (note_h + note_pad*2)
sliders_w = notes_w
sliders_h = slider_h + slider_pad*2
controls_w = control_w * control_num
controls_h = control_h
window_w = notes_w
window_h = notes_h + sliders_h + controls_h
slider_w = (window_w - slider_pad*2) / slider_num
notes_x = 0
# notes_y = sliders_h
notes_y = window_h // 2 - notes_h // 2 
sliders_x = slider_pad
sliders_y = slider_pad
controls_x = (window_w - controls_w) / 2
controls_y = notes_h + sliders_h

#Global variables
prev_mouse_pos = None
mouse_pressed = 0
cur_slider_ix = 0
cur_control_ix = 0
volume = 3000
instrument = 0
needs_update = True
cur_params = np.zeros((num_params,), dtype=np.float32)
cur_notes = np.zeros((num_measures, note_h, note_w), dtype=np.uint8)
cur_controls = np.array(control_inits, dtype=np.float32)

#Setup audio stream
audio = pyaudio.PyAudio()
audio_notes = []
audio_time = 0
note_time = 0
note_time_dt = 0
audio_reset = False
audio_pause = False

cur_playing = []



#Keras
print "Loading Keras..."
import os
os.environ['THEANORC'] = "./" + device + ".theanorc"
os.environ['KERAS_BACKEND'] = "theano"
import theano
print "Theano Version: " + theano.__version__
import keras
print "Keras Version: " + keras.__version__
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Activation, Dropout, Flatten, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.noise import GaussianNoise
from keras.layers.local import LocallyConnected2D
from keras.optimizers import Adam, RMSprop, SGD
from keras.regularizers import l2
from keras.losses import binary_crossentropy
from keras.layers.advanced_activations import ELU
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from keras import backend as K
K.set_image_data_format('channels_first')

print "Loading Encoder..."
model = load_model(dir_name + 'model.h5')
enc = K.function([model.get_layer('encoder').input, K.learning_phase()],
                 [model.layers[-1].output])
enc_model = Model(inputs=model.input, outputs=model.get_layer('pre_encoder').output)

print "Loading Statistics..."
means = np.load(dir_name + sub_dir_name + 'means.npy')
evals = np.load(dir_name + sub_dir_name + 'evals.npy')
evecs = np.load(dir_name + sub_dir_name + 'evecs.npy')
stds = np.load(dir_name + sub_dir_name + 'stds.npy')

# print "Loading Songs..."
y_samples = None
y_lengths = None
# y_samples = np.load('samples.npy')
# y_lengths = np.load('lengths.npy')





def audio_callback(in_data, frame_count, time_info, status):
    global audio_time
    global audio_notes
    global audio_reset
    global note_time
    global note_time_dt

    #Check if needs restart
    if audio_reset:
        audio_notes = []
        audio_time = 0
        note_time = 0
        note_time_dt = 0
        audio_reset = False
    
    #Check if paused
    if audio_pause and status is not None:
        data = np.zeros((frame_count,), dtype=np.float32)
        return (data.tobytes(), pyaudio.paContinue)
    
    #Find and add any notes in this time window
    cur_dt = note_dt
    while note_time_dt < audio_time + frame_count:
        measure_ix = note_time / note_h
        if measure_ix >= num_measures:
            break
        note_ix = note_time % note_h
        notes = np.where(cur_notes[measure_ix, note_ix] >= note_thresh)[0]
        for note in notes:
            # freq = 2 * 38.89 * pow(2.0, note / 12.0) / sample_rate
            freq = note
            audio_notes.append((note_time_dt, freq))
        note_time += 1
        note_time_dt += cur_dt
            
    #Generate the tones
    data = np.zeros((frame_count,), dtype=np.float32)
    for t,f in audio_notes:
        x = np.arange(audio_time - t, audio_time + frame_count - t)
        x = np.maximum(x, 0)

        if instrument == 0:
            w = np.sign(1 - np.mod(x * f, 2))            #Square
        elif instrument == 1:
            w = np.mod(x * f - 1, 2) - 1                 #Sawtooth
        elif instrument == 2:
            w = 2*np.abs(np.mod(x * f - 0.5, 2) - 1) - 1 #Triangle
        elif instrument == 3:
            w = np.sin(x * f * math.pi)                  #Sine
        
        # w[x == 0] = 0
        # w *= volume * np.exp(-x*note_decay)
        # data += w

    ##################
        if f not in cur_playing:
            player.note_on(f + 12, 127, 1)
            cur_playing.append(f)

    for i in cur_playing:
        not_found = True
        for j in audio_notes:
            if i == j[1]:
                not_found = False
                break
        if not_found:
            cur_playing.remove(i)
            player.note_off(i + 12)
    ##################

    # data = np.clip(data, -32000, 32000).astype(np.int16)
    data = np.zeros((frame_count,), dtype=np.float32)

    #Remove notes that are too old
    audio_time += frame_count
    audio_notes = [(t,f) for t,f in audio_notes if audio_time < t + note_duration]
    
    #Reset if loop occurs
    if note_time / note_h >= num_measures:
        audio_time = 0
        note_time = 0
        note_time_dt = 0
        audio_notes = []
    
    #Return the sound clip
    return (data.tobytes(), pyaudio.paContinue)

def start_pygame():
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((window_w, window_h))        
    # screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    notes_surface = screen.subsurface((notes_x, notes_y, notes_w, notes_h))
    pygame.display.set_caption('MusicEdit')
    font = pygame.font.SysFont("monospace", 15)

def stop_pygame():
    pygame.quit()

def start_audio():
    audio = pyaudio.PyAudio()
    return audio

def start_audio_stream():
    audio_stream = audio.open(
        format=audio.get_format_from_width(2),
        channels=1,
        rate=sample_rate,
        output=True,
        stream_callback=audio_callback)
    audio_stream.start_stream()
    return audio_stream

def start_music_player():
    pygame.midi.init()
    player = pygame.midi.Output(0)
    player.set_instrument(115, 1)
    return player

def stop_audio(audio):
    audio.terminate()

def stop_audio_stream(stream):
    audio_stream.stop_stream()
    audio_stream.close()

def stop_music_player(player):
    player.close()

def start_music():
    global audio
    global audio_stream
    global player

    start_pygame()
    audio = start_audio()
    audio_stream = start_audio_stream()
    player = start_music_player()

def stop_music():
    global audio
    global audio_stream
    global player

    stop_music_player(player)
    stop_audio_stream(audio_stream)
    stop_audio(audio)
    stop_pygame()





from video import *


def main():
    global audio_pause
    global needs_update
    global cur_params
    global evals
    global evecs
    global means
    global use_pca
    global cur_notes
    global enc


    start_music()
    open_window()

    instrument = 0
    instruments = [4, 8, 26, 28, 34, 38, 39, 53, 54, 108, 118]
    player.set_instrument(instruments[instrument], 1)

    video = 0
    num_videos = 3
    vid1, vid2 = load_videos(video)
    i1 = 0
    i2 = 0
    len1 = len(vid1)
    len2 = len(vid2)

    running = True
    while running:
        start_time = time.time()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            elif event.type == pygame.KEYDOWN:
                if event.key in {pygame.K_ESCAPE, pygame.K_q}:
                    running = False
                elif event.key == pygame.K_UP:
                    instrument = (instrument + 1) % len(instruments)
                    player.set_instrument(instruments[instrument], 1)
                elif event.key == pygame.K_DOWN:
                    instrument = (instrument - 1) % len(instruments)
                    player.set_instrument(instruments[instrument], 1)
                elif event.key == pygame.K_RIGHT:
                    video = (video + 1) % num_videos
                    vid1, vid2 = load_videos(video)
                    i1 = 0
                    i2 = 0
                    len1 = len(vid1)
                    len2 = len(vid2)
                elif event.key == pygame.K_LEFT:
                    video = (video - 1) % num_videos
                    vid1, vid2 = load_videos(video)
                    i1 = 0
                    i2 = 0
                    len1 = len(vid1)
                    len2 = len(vid2)

        if LEAP:
            # update music
            needs_update, params = getMusicPCAFromLeap(leapController)
            if needs_update:
                cur_params[:len(params)] = params
                if use_pca:
                    x = means + np.dot(cur_params * evals, evecs)
                else:
                    x = means + stds * cur_params
                x = np.expand_dims(x, axis=0)
                y = enc([x, 0])[0][0]
                cur_notes = (y * 255.0).astype(np.uint8)
                needs_update = False
            # update video
            indices = getVideoIndicesFromLeap(leapController)
            if indices is None:
                continue
            else:
                i1,i2 = indices
                i1 = i1 % len1
                i2 = i2 % len2
        else:
            # update music (TODO)
            # update video
            i1,i2 = (i1+1)%len1,(i2+1)%len2

        cv2.imshow("window", blend(vid1[i1], vid2[i2]))
        runtime = time.time() - start_time
        if runtime < 0.02:
            time.sleep(0.02 - runtime)

    stop_music()
    close_window()

if __name__ == '__main__':
    main()