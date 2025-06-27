# Weed_Detection Using YOLO with Semi-Supervised Learning
This repository has a weed detection model that uses YOLO, semi-supervised learning, and both labeled and unlabeled data. The goal is to find and locate weeds in farm fields to use less herbicide and help predict better crop yields.

# Dataset
1. Labeled Dataset: 200 images of sesame crops and weeds.
2. Unlabeled Dataset: 1000 similar images.
3. Test Dataset: 100 images with annotations. 

# Approach
The approach combines supervised learning using labeled data and semi-supervised learning using unlabeled data to enhance model performance. We applied the following techniques:

# 1. Data Preprocessing & Augmentation:

(a) **Rotation:** 90°, 180°, and 270° to diversify weed orientations.
(b) **Horizontal Flipping:** To increase generalization.
(c) **Bounding Box Adjustments:** After augmentations to maintain correct localization.

# 2. YOLO11m Training:

  (a) Training: 50 epochs using labeled data with a batch size of 16, learning rate of 5e-5, and image size of 640x640.
  (b) GPU Acceleration: Using CUDA for faster training. 

# 3. Semi-Supervised Learning:

  (a) Consistency Regularization: Applied weak and strong augmentations to ensure model stability.
  (b) Pseudo Labeling: Generated pseudo labels for unlabeled data based on high-confidence predictions.
  (c) FixMatch Algorithm: Used pseudo-labeled data for further model training.
  (d) Mean Teacher Model: Utilized a Teacher model with EMA to guide learning.

# 4. Inference:
  (a) Test Inference: Performed inference with a confidence threshold of 0.5 to detect and localize weeds.
  (b)Visualization: Bounding boxes drawn around detected weeds and crops for qualitative assessment.

# Evaluation Metrics

The performance of the model was evaluated using:

 (a) Precision and Recall for detecting weeds.
 (b) F1-Score to measure the harmonic mean of precision and recall.
 (c) Mean Average Precision (mAP@[.5:.95]) to evaluate the model across different IoU thresholds.
 (d) Combined Metric: 0.5 * (F1-Score) + 0.5 * (mAP@[.5:.95]).

# Results:

 The model achieved strong performance with the following:

  (a) High precision and recall in weed detection.
  (b) Lower mAP scores due to discrepancies between ground truth labels and model predictions.
  (c) Improved weed localization and detection, even with partial or missing ground truth annotations.

# Challenges:

  (a) Discrepancy in Ground Truth Labels: The model's predictions were often more accurate in terms of localization than the ground truth, leading to lower precision and recall but better detection.
  (b) The model demonstrated the potential for improving annotation quality in future datasets by identifying missing or misclassified weeds.

# Inference:

 The final model was evaluated on a set of test images, where it successfully detected and  localized weeds. Bounding boxes closely aligned with ground truth, and the model performed well under varying  environmental conditions.


## Model Performance Comparison

Below is a comparison of the performance metrics for different model variations during training and testing:

| **Model**                       | **Precision** | **Recall** | **mAP50** | **mAP50–95** | **Fitness** | **F1 Score** | **Combined Metric** |
|--------------------------------|---------------|------------|-----------|--------------|-------------|--------------|----------------------|
| *Base Model*                   | 0.8619        | 0.7655     | 0.8455    | 0.5005       | 0.5350      | 0.8108       | 0.6557               |
| Consistency Regularization     | 0.8078        | 0.8481     | 0.8409    | 0.5150       | 0.5476      | 0.8275       | 0.6712               |
| Pseudo Labeling                | 0.8027        | 0.8212     | 0.8408    | 0.5157       | 0.5482      | 0.8118       | 0.6637               |
| FixMatch                       | 0.8681        | 0.8010     | 0.8541    | 0.5392       | 0.5707      | 0.8332       | 0.6862               |
| Mean Teacher (100 epochs)      | 0.8426        | 0.8596     | 0.8747    | 0.5755       | 0.6054      | 0.8510       | 0.7133               |
| Mean Teacher (200 epochs)      | 0.9175        | 0.8505     | 0.9451    | 0.6956       | 0.7206      | 0.8827       | 0.7892               |

## Conclusion:

This weed detection model, which integrates supervised and semi-supervised learning techniques, proves effective for real-world agricultural use. It offers a robust solution for detecting and localizing weeds, ultimately supporting more precise and efficient crop management.

---

## Requirements:

- Python 3.x  
- PyTorch  
- YOLOv8 or YOLO11m (Ultralytics)  
- OpenCV  
- CUDA (for GPU acceleration)

---

## Usage:

```bash
# 1. Clone the repository
git clone https://github.com/your-repo/weed-detection.git

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model
python train.py --data dataset.yaml --epochs 50 --batch-size 16 --img-size 640

# 4. Run inference on test images
python infer.py --weights path/to/model.pt --source path/to/test/images

