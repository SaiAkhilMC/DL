from google.colab import drive
drive.mount('/content/drive')

!pip install ultralytics

from ultralytics import YOLO

model=YOLO("yolov8m.pt")

print("Model architcture: \n\n", model)
print("type(model)",type(model))

import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

test_image_path='/content/drive/MyDrive/Deep Learning/cat_dog.jpg'
test_image=cv2.imread(test_image_path)
print("image_shape:",test_image.shape)
plt.imshow(test_image)
plt.axis("off")

results=model.predict(test_image)
print(len(results))
print(results)

result= results[0]
bounding_box_predictions=result.boxes
print(bounding_box_predictions)

bounding_box_xyxy_coordinates=bounding_box_predictions.xyxy
print(type(bounding_box_xyxy_coordinates))

bounding_box_xyxy_coordinates_list=bounding_box_xyxy_coordinates.tolist()
print(bounding_box_xyxy_coordinates_list)

from google.colab.patches import cv2_imshow

for box_index in range(len(bounding_box_xyxy_coordinates_list)):
  print("Showing Box ",box_index+1 ,"\n")
  x1,y1,x2,y2=bounding_box_xyxy_coordinates_list[box_index]
  x1=int(x1)
  y1=int(y1)
  x2=int(x2)
  y2=int(y2)
  cv2.rectangle(test_image,(x1,y1),(x2,y2),(0,255,0),2)
  cv2_imshow(test_image)
  print("\n\n")

