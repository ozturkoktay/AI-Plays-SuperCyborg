import numpy as np
from grabscreen import grab_screen
import cv2
import time
from directkeys import PressKey,ReleaseKey, W, A, S, D, F
from alexnet import alexnet
from getkeys import key_check

import random

WIDTH = 80
HEIGHT = 60
LR = 1e-3
EPOCHS = 8
MODEL_NAME = 'pycyborg-{}-{}-{}-epochs.model'.format(LR, 'alexnetv2', EPOCHS)
t_time = 0.09

def straight():
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(S)

def fire():
    PressKey(F)
    ReleaseKey(S)
    ReleaseKey(A)
    ReleaseKey(W)
    time.sleep(t_time)

def left():
    PressKey(A)
    #ReleaseKey(W)
    ReleaseKey(D)
    #ReleaseKey(A)
    time.sleep(t_time)
    ReleaseKey(A)

def right():
    PressKey(D)
    ReleaseKey(A)
    #ReleaseKey(W)
    #ReleaseKey(D)
    time.sleep(t_time)
    ReleaseKey(D)

    
model = alexnet(WIDTH, HEIGHT, LR)
model.load(MODEL_NAME)

def main():
    last_time = time.time()
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    paused = False
    while(True):
        
        if not paused:
            # 800x600 windowed mode
            #screen =  np.array(ImageGrab.grab(bbox=(0,40,800,640)))
            screen = grab_screen(region=(0,40,800,640))
            print('loop took {} seconds'.format(time.time()-last_time))
            last_time = time.time()
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (80,60))

            prediction = model.predict([screen.reshape(WIDTH,HEIGHT,1)])[0]
            print(prediction)

            turn_thresh = .75
            fwd_thresh = 0.70

            if prediction[1] > fwd_thresh:
                fire()
            elif prediction[0] > turn_thresh:
                left()
            elif prediction[2] > turn_thresh:
                right()
            else:
                straight()

        keys = key_check()

        if 'T' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)
                time.sleep(1)

main()       





        
