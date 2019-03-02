import numpy as np
import os
import cv2
import time

"""
Train initial background samples
"""
def initial_background(I_gray, N):
    print('start training initial model')
    I_pad = np.pad(I_gray, 1, 'symmetric')
    height = I_pad.shape[0]
    width = I_pad.shape[1]
    # N samples
    samples = np.zeros((height,width,N))
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            for n in range(N):
                x, y = 0, 0
                while(x == 0 and y == 0):
                    x = np.random.randint(-1, 1)
                    y = np.random.randint(-1, 1)
                ri = i + x
                rj = j + y
                samples[i, j, n] = I_pad[ri, rj]
    samples = samples[1:height-1, 1:width-1]
    return samples

"""
Detector and upgrade of background samples
""" 
def vibe_detection(I_gray, samples, _min, N, R):
    
    height = I_gray.shape[0]
    width = I_gray.shape[1]
    # create empty mask = 0, update foreground pixel = 255 later.
    segMap = np.zeros((height, width)).astype(np.uint8)

    """
    New version: update of sample background (speed optimized by removing the for loop)
    """
    mask_count = np.zeros_like(segMap) 
    for index in range(N):
        dist = np.abs(I_gray - samples[:,:,index])
        [x,y] = np.where(dist<R)
        mask = np.zeros_like(I_gray)
        mask[x,y] = 1
        mask_count += mask
    
    # mask_count > _min refers to background
    [x,y] = np.where(mask_count>_min)
    # perturb: list to store all possible neighbours perturbation
    perturb = [[-1,1],[-1,0],[-1,-1],[0,1],[0,-1],[1,1],[1,0],[1,-1]]

    len_x = len(x)

    # Generate index of random frames (chosen from N frames in the model). 
    # The length of random frames is the same as the number of identified background pixels.
    index_frame = np.random.randint(N,size=(len_x,))
    # For each random frame, generate a number r from 0 to N-1. If r!=0, will not update the model.
    r = np.random.randint(N,size=(len_x,))
    x[r!=0] = 0
    y[r!=0] = 0
    # Update the sample and the pixel value
    samples[x,y,index_frame] = I_gray[x,y]

    # Generate a vector "perturb_vec", which indicates the perturb applied to each background pixel. 
    # For example, (refer to "perturb = [[-1,1],[-1,0],[-1,-1],[0,1],[0,-1],[1,1],[1,0],[1,-1]]"), perturb_vec[k] = 2 means apply [-1,-1] to k-th background pixel [x,y].
    perturb_vec = np.random.randint(8,size=len_x)
    perturb_list = np.zeros((len_x,2))
    for row in range(len_x):
        perturb_list[row,:] = perturb[perturb_vec[row]]
    perturb_list = perturb_list.astype(int)
    perturb_x = np.reshape(perturb_list[:,0],(len_x,))
    perturb_y = np.reshape(perturb_list[:,1],(len_x,))
    # Generate index of random frames (chosen from N frames in the model). 
    # The length of random frames is the same as the number of identified background pixels.
    index_frame = np.random.randint(0, N-1)
    # During the update, exclude the boundary of the image.
    x[x==height-1] = height-2
    x[x==0] = 1
    y[y==width-1] = width - 2
    y[y==0] = 1
    # For each random frame, generate a number r from 0 to N-1. If r!=0, will not update the model.
    r = np.random.randint(N,size=len_x)
    x[r!=0] = 0
    y[r!=0] = 0
    perturb_x[r!=0] = 0
    perturb_y[r!=0] = 0
    samples[x+perturb_x,y+perturb_y,index_frame] = I_gray[x,y]

    # mask_count <= _min refers to foreground
    [x,y] = np.where(mask_count<=_min)
    segMap[x,y] = 255

    """
    Old version: upgrade of sample background by original github code. It applied a for loop, which makes the algo slow.
    """
    # for i in range(height):
    #     for j in range(width):
    #         count, index, dist = 0, 0, 0
    #         while count < _min and index < N:
    #             dist = np.abs(I_gray[i,j] - samples[i,j,index])
    #             if dist < R:
    #                 count += 1
    #             index += 1
    #         if count >= _min:
    #             r = np.random.randint(0, N-1)
    #             if r == 0:
    #                 r = np.random.randint(0, N-1)
    #                 samples[i,j,r] = I_gray[i,j]
    #             r = np.random.randint(0, N-1)
    #             if r == 0:
    #                 x, y = 0, 0
    #                 while(x == 0 and y == 0):
    #                     x = np.random.randint(-1, 1)
    #                     y = np.random.randint(-1, 1)
    #                 r = np.random.randint(0, N-1)
    #                 ri = i + x
    #                 rj = j + y
    #                 try:
    #                     samples[ri, rj, r] = I_gray[i, j]
    #                 except:
    #                     pass
    #         else:
    #             segMap[i, j] = 255

    return segMap, samples
    
# rootDir = r'data/input'
# image_file = os.path.join(rootDir, os.listdir(rootDir)[0])
# image = cv2.imread(image_file, 0)

N = 20 # N samples in the model
R = 20 # threshold for foreground identification
_min = 2 # minimum matches to identify background

cap = cv2.VideoCapture(0)

count = 0

start = time.time()

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # resize the frame
    frame = cv2.resize(frame, (480, 360), interpolation = cv2.INTER_CUBIC)
    rows,cols, channels = frame.shape
    # change frame to gray level
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    time_taken = time.time()-start

    if time_taken < 2:
        continue
    if count == 0:
        # build initial model
        samples = initial_background(gray,N)
        count += 1
    else:
        count += 1

    # predict the foreground and update the model
    segMap, samples = vibe_detection(gray, samples, _min, N, R)

    red_mask = np.zeros_like(frame)
    red_mask[segMap==255] = (255,0,0)
    res = cv2.addWeighted(frame,1.0,red_mask,0.7,0)

    cv2.imshow('segMap',segMap)
    cv2.imshow('red foreground',res)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()