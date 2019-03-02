import numpy as np
import cv2
import time
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

start = time.time() #starting time
start0 = start

count = 0 # the index of image in the trained set.
time_taken = 0 # initialize time_taken variable.
flag = 1 # flag = 0 when model is trained.
path_video = "C:/Users/zyt/Desktop/BG_subtraction/video.avi"
out = cv2.VideoWriter(path_video,cv2.VideoWriter_fourcc('M','J','P','G'), 10, (640,480))

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # ouptut size of the frame
    rows, cols = frame.shape[:2]
    # denote the total number of frames used to constructed the background model
    total_frame = 200

    # start saving images after 2s
    if time_taken >= 2 and count < total_frame:
        if count == 0:
            # initialization of blue, green and red channel for the background frames
            tmp = frame.copy() # save another copy of frame, since I don't want to change the value of frame.
            tmp = np.float32(tmp) # frame dtype is np.uint8, tmp dtype if float32.
            bg_blue = np.zeros((rows,cols,total_frame))
            bg_green = np.zeros((rows,cols,total_frame))
            bg_red = np.zeros((rows,cols,total_frame))
            bg_blue[:,:,count] = tmp[:,:,0] # the first layer of bg_blue
            bg_green[:,:,count] = tmp[:,:,1] # the first layer of bg_green
            bg_red[:,:,count] = tmp[:,:,2] # the first layer of bg_red
        else:
            # save the next each consecutive frame
            tmp = frame.copy()
            tmp = np.float32(tmp)
            bg_blue[:,:,count] = tmp[:,:,0] # the layer numbered 'count' of bg_blue
            bg_green[:,:,count] = tmp[:,:,1]
            bg_red[:,:,count] = tmp[:,:,2]
        # increment the count by 1
        count += 1

    time_taken = time.time()-start

    # train the model with the input images set
    if time_taken > 2 and flag == 1 and count == total_frame: # count == total_frame, have saved all the images for training purpose.
        bg_blue_std = np.zeros((rows,cols),np.float32)
        bg_blue_mean = np.zeros((rows,cols),np.float32)
        bg_green_std = np.zeros((rows,cols),np.float32)
        bg_green_mean = np.zeros((rows,cols),np.float32)
        bg_red_std = np.zeros((rows,cols),np.float32)
        bg_red_mean = np.zeros((rows,cols),np.float32)
        # go through all the rows and cols, to update the standard deviation and the mean.
        for i in range(rows):
            if i%10 == 0 or i==rows-1:
                print('trained percentage:',round(100*i/(rows-1),2),'%')
            for j in range(cols):
                # set a threshold for the minimum standard deviation, to make the algorithm most robust to lighting condition change.
                # if we don't want to deal with the noise, then let threshold = 0
                threshold = 6
                bg_blue_std[i,j] = max(threshold,np.std(bg_blue[i,j,:]))

                bg_blue_mean[i,j] = np.mean(bg_blue[i,j,:])

                bg_green_std[i,j] = max(threshold,np.std(bg_green[i,j,:]))

                bg_green_mean[i,j] = np.mean(bg_green[i,j,:])

                bg_red_std[i,j] = max(threshold,np.std(bg_red[i,j,:]))

                bg_red_mean[i,j] = np.mean(bg_red[i,j,:])

        flag = 0
        count += 1
        print('Model is formed.','Time taken:',time.time()-start)
        start0 = time.time()

    # after the model is formed, start analyzing the frame, separate that into foreground and background.
    if count == total_frame+1:
        mask = np.ones((rows,cols),np.uint8)*255 # 255 for background

        # if the pixel value fall out of 5 standard deviation of the mean from the trained model, it is regarded as the foreground.
        std_range = 5

        blue_upper = cv2.add( np.uint8(bg_blue_mean), np.uint8(std_range*bg_blue_std))
        [x,y] = np.where(frame[:,:,0]>blue_upper)
        mask[x,y] = 0 # 0 for foreground
        blue_lower = cv2.subtract( np.uint8(bg_blue_mean), np.uint8(std_range*bg_blue_std))
        [x,y] = np.where(frame[:,:,0]<blue_lower)
        mask[x,y] = 0

        green_upper = cv2.add( np.uint8(bg_green_mean), np.uint8(std_range*bg_green_std))
        [x,y] = np.where(frame[:,:,1]>green_upper)
        mask[x,y] = 0
        green_lower = cv2.subtract( np.uint8(bg_green_mean), np.uint8(std_range*bg_green_std))
        [x,y] = np.where(frame[:,:,1]<green_lower)
        mask[x,y] = 0

        red_upper = cv2.add( np.uint8(bg_red_mean), np.uint8(std_range*bg_red_std))
        [x,y] = np.where(frame[:,:,2]>red_upper)
        mask[x,y] = 0
        red_lower = cv2.subtract( np.uint8(bg_red_mean), np.uint8(std_range*bg_red_std))
        [x,y] = np.where(frame[:,:,2]<red_lower)
        mask[x,y] = 0

        mask_tmp = mask.copy() # copy for the same purpose
        mask[mask_tmp==255] = 0 # invert mask, 0 for background, 255 foreground
        mask[mask_tmp==0  ] = 255

        # create a light blue mask overlay on the foreground.
        mask_frame = np.zeros_like(frame)
        mask_frame[mask==255] = (255,0,0)
        frame = cv2.addWeighted(frame,1.0,mask_frame,0.5,0)
                
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if count == total_frame+1:
        cv2.imshow('mask',mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time_taken0 = time.time() - start0
    print('time taken:',time_taken0)
    if time_taken0 > 10 and time_taken0 < 30:
        out.write(frame)
        
# When everything done, release the capture
cap.release()
out.release()
cv2.destroyanyWindows()
