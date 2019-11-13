2# -*- coding: utf-8 -*-

import cv2
import operator
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.signal import argrelextrema
import time
import argparse
from detector import FaceDetector
from nms import non_max_suppression
from skimage.transform import pyramid_reduce
from skimage.transform import pyramid_gaussian

print(sys.executable)
#Setting fixed threshold criteria
USE_THRESH = False
#fixed threshold value
THRESH = 0.6
#Setting fixed threshold criteria
USE_TOP_ORDER = False
#Setting local maxima criteria
USE_LOCAL_MAXIMA = True
#Number of top sorted frames
NUM_TOP_FRAMES = 20

#Video path of the source file
videopath = sys.argv[1]
#Directory to store the processed frames
dir = sys.argv[2]
#smoothing window size
len_window = int(sys.argv[3])

def sliding_window(image, step_size, window_size):
	""" Slide a window across an image. """
	for y in range(0, image.shape[0], step_size):
		for x in range(0, image.shape[1], step_size):
			# Create window
			yield x, y, image[y:y+window_size[1], x:x+window_size[0]]


def smooth(x, window_len=13, window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    import numpy as np    
    t = np.linspace(-2,2,0.1)
    x = np.sin(t)+np.random.randn(len(t))*0.1
    y = smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string   
    """
    print(len(x), window_len)
    if x.ndim != 1:
        raise ValueError  #, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError  #, "Input vector needs to be bigger than window size."

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError  #, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s = np.r_[2 * x[0] - x[window_len:1:-1],
              x, 2 * x[-1] - x[-1:-window_len:-1]]
    #print(len(s))

    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = getattr(np, window)(window_len)
    y = np.convolve(w / w.sum(), s, mode='same')
    return y[window_len - 1:-window_len + 1]

#Class to hold information about each frame


class Frame:
    def __init__(self, id, frame, value):
        self.id = id
        self.frame = frame
        self.value = value

    def __lt__(self, other):
        if self.id == other.id:
            return self.id < other.id
        return self.id < other.id

    def __gt__(self, other):
        return other.__lt__(self)

    def __eq__(self, other):
        return self.id == other.id and self.id == other.id

    def __ne__(self, other):
        return not self.__eq__(other)


def rel_change(a, b):
   x = (b - a) / max(a, b)
   print(x)
   return x


print("Video :" + videopath)
print("Frame Directory: " + dir)


cap = cv2.VideoCapture(str(videopath))

detector = FaceDetector('../model/weights.pkl')

curr_frame = None
prev_frame = None

frame_diffs = []
frames = []
ret, frame = cap.read()
i = 1

while(ret):

    src = frame

    # Start detection process
    boxes = []
    scale_factor = 1
    for img in pyramid_gaussian(np.copy(src), downscale=2, multichannel=True):
            # Break if the scaled image is smaller than the window 
            if img.shape[0] < window_size[0] or img.shape[1] < window_size[1]:
                    break

            for (x, y, window) in sliding_window(img, step_size, window_size):
                    # Continue if window does not meet desired size
                    if window.shape[0] < window_size[0] or window.shape[1] < window_size[1]:
                            continue 

                    # Detect a face
                    confidence = detector.detect(window)
                    if confidence > 0:
                            x1 = x * scale_factor
                            y1 = y * scale_factor
                            x2 = x1 + window_size[1] * scale_factor 
                            y2 = y1 + window_size[0] * scale_factor
                            boxes.append((x1, y1, x2, y2, confidence))

            # Keep track of current scale factor
            scale_factor *= 2

    flag =  false
    # Check if there are detections
    if len(boxes) == 0:
        #print('No faces detected')
        #exit()
        g=4
    else:
        flag=true
        luv = cv2.cvtColor(frame, cv2.COLOR_BGR2LUV)
        curr_frame = luv
        if curr_frame is not None and prev_frame is not None:
            #logic here
            diff = cv2.absdiff(curr_frame, prev_frame)
            count = np.sum(diff)
            frame_diffs.append(count)
            frame = Frame(i, frame, count)
            frames.append(frame)
        prev_frame = curr_frame
        
    i = i + 1
    ret, frame = cap.read()

cap.release()
#cv2.destroyAllWindows()

if USE_TOP_ORDER:
    # sort the list in descending order
    frames.sort(key=operator.attrgetter("value"), reverse=True)
    for keyframe in frames[:NUM_TOP_FRAMES]:
        name = "frame_" + str(keyframe.id) + ".jpg"
        cv2.imwrite(dir + "/" + name, keyframe.frame)

if USE_THRESH:
    print("Using Threshold")
    for i in range(1, len(frames)):
        if (rel_change(np.float(frames[i - 1].value), np.float(frames[i].value)) >= THRESH):
            #print("prev_frame:"+str(frames[i-1].value)+"  curr_frame:"+str(frames[i].value))
            name = "frame_" + str(frames[i].id) + ".jpg"
            cv2.imwrite(dir + "/" + name, frames[i].frame)


if USE_LOCAL_MAXIMA:
    print("Using Local Maxima")
    diff_array = np.array(frame_diffs)
    sm_diff_array = smooth(diff_array, len_window)
    frame_indexes = np.asarray(argrelextrema(sm_diff_array, np.greater))[0]
    for i in frame_indexes:
        name = "frame_" + str(frames[i - 1].id) + ".jpg"
        #print(dir+name)
        cv2.imwrite(dir + name, frames[i - 1].frame)

'''
plt.figure(figsize=(40, 20))
plt.locator_params(numticks=100)
plt.stem(sm_diff_array)
plt.savefig(dir + 'plot.png')
'''
