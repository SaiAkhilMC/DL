from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
import torchvision
from torchvision.ops import nms
import matplotlib.patches as patches

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

COCO_INSTANCE_CATEGORY_NAMES = ["__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
                                "train", "truck", "boat", "traffic light", "fire hydrant", "N/A", "stop sign",
                                "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                                "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A", "N/A",
                                "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                                "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
                                "bottle", "N/A", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
                                "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
                                "donut", "cake", "chair", "couch", "potted plant", "bed", "N/A", "dining table",
                                "N/A", "N/A", "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                                "microwave", "oven", "toaster", "sink", "refrigerator", "N/A", "book",
                                "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

test_image_path = '/content/drive/MyDrive/Deep Learning/cat_dog.jpg'
test_image = Image.open(test_image_path)
transform = T.Compose([T.ToTensor()])
test_image_tensor = transform(test_image)

with torch.no_grad():
    predictions = model([test_image_tensor])

if 'boxes' in predictions[0]:
    prediction = predictions[0]

    bounding_boxes = prediction['boxes'].detach().numpy()
    class_labels = prediction['labels'].numpy()
    prediction_probabilities = prediction['scores'].detach().numpy()

    test_image = cv2.imread(test_image_path)
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    for i, box in enumerate(bounding_boxes):
        x1, y1, x2, y2 = map(int, box)
        category = COCO_INSTANCE_CATEGORY_NAMES[class_labels[i]]
        score = prediction_probabilities[i]
        cv2.rectangle(test_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(test_image, f"{category} {score:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    plt.imshow(test_image)
    plt.show()

    nms_thresholds = np.linspace(0, 1, 20)
    for threshold in nms_thresholds:
        keep_indices = nms(torch.tensor(bounding_boxes), torch.tensor(prediction_probabilities), threshold)
        filtered_boxes = bounding_boxes[keep_indices]

        fig, ax = plt.subplots(1)
        ax.imshow(test_image)

        for box in filtered_boxes:
            x, y, w, h = box
            rect = patches.Rectangle((x, y), w - x, h - y, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        plt.title(f"NMS Threshold: {threshold}")
        plt.show()
else:
    print("No predictions found.")

