from google.colab import drive
drive.mount('/content/drive')

import cv2
from google.colab.patches import cv2_imshow

image_path = '/content/drive/My Drive/VAP/scene.png'

image = cv2.imread(image_path)

if image is not None:
    print("type(image): \t", type(image), "\n")
    print("image.shape: \t", image.shape, "\n")
    print("Image width: \t", image.shape[0], "\n")
    print("Image height: \t", image.shape[1], "\n")
    print("Image channels: \t", image.shape[2], "\n")
    print("Now displaying the image \n\n")
    cv2_imshow(image)
else:
    print("Error: Image not found or could not be loaded.")

image_path = '/content/drive/My Drive/VAP/scene.png'

image = cv2.imread(image_path)

if image is not None:
    print("Original resolution: \t", image.shape, "\n")
    cv2_imshow(image)

    image_resized = cv2.resize(image, (400, 400))
    print("\n Resized resolution: \t", image_resized.shape, "\n")
    cv2_imshow(image_resized)

else:
    print("Error: Image not found or could not be loaded.")

image_rgb = image_resized

image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2HSV)

print("Image in the RGB color space \n")
cv2_imshow(image_rgb)

print("\n\nImage in the HSV color space \n")
cv2_imshow(image_hsv)

import numpy as np

image_normalized = image_rgb / 255.0

C = 1 - image_normalized[:, :, 2]
M = 1 - image_normalized[:, :, 1]
Y = 1 - image_normalized[:, :, 0]

K = np.min([C, M, Y], axis=0)

image_cmyk = np.dstack((C, M, Y, K))

image_cmyk = (image_cmyk * 255).astype(np.uint8)

print("Image in the RGB color space \n")
cv2_imshow(image_rgb)

print("\n\nImage in the CMYK color space \n")
cv2_imshow(image_cmyk)

image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)

print("Image in the RGB color space \n")
cv2_imshow(image_rgb)

print("\n\nImage in the grayscale color space \n")
cv2_imshow(image_gray)

print("Number of channels: \t", image_gray.shape, "\n")

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_color = (0, 0, 255)
font_thickness = 1

cv2.putText(image_rgb, "Hello world", (30, 30), font, font_scale, font_color, font_thickness)
cv2.circle(image_rgb, (200,200), 5, font_color, font_thickness)

cv2_imshow(image_rgb)

image_rgb = image_rgb

border_size1 = 5
border_size2 = 10
border_color = (255, 0, 0)
image_with_border = cv2.copyMakeBorder(image_rgb, border_size1, border_size1, border_size2, border_size2, cv2.BORDER_CONSTANT, value=border_color)
cv2_imshow(image_with_border)

print("\n\n\n")

cv2_imshow(image_rgb)

import os

folder_path = '/content/drive/My Drive/VAP/'

image_data_list = []

file_list = os.listdir(folder_path)
print("The files listed in the folder are: \n\n", file_list, "\n")

for filename in file_list:

    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):

        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)

        if image is not None:
            image_resized = cv2.resize(image, (640, 480))
            image_data_list.append(image_resized)

print("Number of images read: \t", len(image_data_list), "\n")

image_data_array = np.array(image_data_list)
print("Shape of the image data array:", image_data_array.shape)

for i in range(image_data_array.shape[0]):
    image_to_plot = image_data_array[i]
    print("Showing image: ", (i+1), "\n")
    cv2_imshow(image_to_plot)
    print("\n\n")

image_contrast_enhanced = cv2.equalizeHist(image_gray)

print("Original Grayscale image: \n")
cv2_imshow(image_gray)

print("\n\n Contrast Enhanced image: \n")
cv2_imshow(image_contrast_enhanced)

print("original image: \n")
cv2_imshow(image_rgb)
print("\n\n")

gamma_values = [0.25, 0.5, 1.0, 1.5, 2.0]

for gamma in gamma_values:
    print("Brightness adjusted using gamma = ", gamma, "\n")
    image_gamma_corrected = np.power(image_rgb / 255.0, gamma) * 255.0
    image_gamma_corrected = np.uint8(image_gamma_corrected)
    cv2_imshow(image_gamma_corrected)
    print("\n\n")

import cv2
from google.colab.patches import cv2_imshow
image_path = '/content/drive/MyDrive/Module 1 - Datasets/Image Shaperning.png'
image = cv2.imread(image_path)
print("Original image: \n")
cv2_imshow(image)

kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
image_sharp = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
cv2_imshow(image_sharp)

import cv2
from google.colab.patches import cv2_imshow
image_path = '/content/drive/MyDrive/Module 1 - Datasets/scene2.jpg'
img = cv2.imread(image_path)
print("Original image: \n")
cv2_imshow(img)

kernel_25 = np.ones((25,25), np.float32) / 625.0
output_kernel = cv2.filter2D(img, -1, kernel_25)
cv2_imshow(output_kernel)

output_blur = cv2.blur(img, (25,25))
output_box = cv2.boxFilter(img, -1, (5,5))
cv2_imshow(output_blur)
cv2_imshow(output_box)

output_gaus = cv2.GaussianBlur(img, (5,5), 0)
cv2_imshow(output_gaus)

output_med = cv2.medianBlur(img, 5)
cv2_imshow(output_med)

output_bil = cv2.bilateralFilter(img, 5, 6, 6)
cv2_imshow(output_bil)


cv2_imshow(output_kernel)
cv2_imshow(output_blur)
cv2_imshow(output_box)
cv2_imshow(output_gaus)
cv2_imshow(output_bil)
cv2_imshow(output_med)

angle = 45

height, width = image_rgb.shape[:2]
center = (width/2, height/2)

rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

rotated_image = cv2.warpAffine(image_rgb, rotation_matrix, (width, height))

print("Original Image \n")
cv2_imshow(image_rgb)
print("\n\n")

print("Rotated Image \n")
cv2_imshow(rotated_image)
print("\n\n")

print("original_image_shape: \t", image_rgb.shape, "\n")
print("Rotated image_shape: \t", rotated_image.shape, "\n")

image_flipped = cv2.flip(image_rgb, -1)

print("Original Image \n")
cv2_imshow(image_rgb)
print("\n\n")

print("Flipped Image \n")
cv2_imshow(image_flipped)
print("\n\n")

new_width = 400
new_height = 200

image_scaled = cv2.resize(image_rgb, (new_width, new_height))

print("Original Image \n")
cv2_imshow(image_rgb)
print("\n\n")
print("Scaled Image \n")
cv2_imshow(image_scaled)
print("\n\n")

import cv2
from google.colab.patches import cv2_imshow

image_path = '/content/drive/MyDrive/Module 1 - Datasets/Morphological operations.jpg'

image = cv2.imread(image_path)
cv2_imshow(image)

kernel2 = np.ones((3, 3), np.uint8)
image_erode2 = cv2.erode(image, kernel2)
cv2_imshow(image_erode2)

kernel3 = np.ones((2,2), np.uint8)
image_dilation = cv2.dilate(image, kernel, iterations=1)
cv2_imshow(image_dilation)

kernel_size = 5
kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

image_opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
cv2_imshow(image_opening)

kernel_size = 5
kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

image_close = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
cv2_imshow(image_close)
