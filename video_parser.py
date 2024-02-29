import cv2
import os
import numpy as np
# should convert the methods to load files in chucks 

def generateFrames(video):
  video = cv2.VideoCapture(video)
  success,image = video.read()
  count = 10000
  images = []

  while success: 
    file = "parsed_video/frame%d.jpg" % count
    print(f"file: {file}")
    images.append(file)

    cv2.imwrite(file,image)     # save frame as JPEG file      
    success,image = video.read()
    print('Read a new frame: ', success)
    count += 1

  print(count)
  return images

def createVideo(path):
  files = os.listdir(path)
  files.sort()

  # codec = cv2.VideoWriter_fourcc('M','J','P','G')
  codec = cv2.VideoWriter_fourcc(*'mp4v')
  fps = 30  
  frame_size = (640, 640)  # (width, height), must match the input image sizes
  out = cv2.VideoWriter('output.avi', codec, fps, frame_size)

  for file in files:
      if file.endswith('.jpg'):
        image_path = os.path.join(path,file)
        frame = cv2.imread(image_path)
        if frame is not None:
          out.write(frame)
          cv2.imshow('frame', frame)
          if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        else:
          print(f'failed to read from {image_path}')

  out.release

  # while(True):
  #   ret, frame = cap.read() 
    
  #   if ret == True:  
  
  #       # Write the frame into the 
  #       # file 'filename.avi' 
  #       out.write(frame) 
  
  #       # Display the frame 
  #       # saved in the file 
  #       cv2.imshow('Frame', frame) 
  
  #       # Press S on keyboard  
  #       # to stop the process 
  #       if cv2.waitKey(1) & 0xFF == ord('s'): 
  #           break
  
  #   # Break the loop 
  #   else: 
  #       break
