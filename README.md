# Analyzing CNN Architectures for Image Classification on CIFAR-10

**Author:** Jithendra Varma Chamarthi  
**Dataset:** CIFAR-10  
**Models Evaluated:** ResNet-18, MobileNet, EfficientNet

---

## Abstract
This report presents a comparative analysis of CNN architectures for CIFAR-10 image classification. By evaluating ResNet-18, MobileNet, and EfficientNet, we investigate the strengths and trade-offs of each model, focusing on accuracy, efficiency, and generalization. Key improvements were made through hyperparameter tuning and data augmentation, with ResNet-18 showing the highest performance.

---

## 1. Introduction

### 1.1 Problem Background
CIFAR-10 consists of 60,000 32x32 images in 10 categories. Despite its low resolution, it serves as a benchmark for CNN evaluation. CNNs detect image features through convolutional layers, making them ideal for complex datasets like CIFAR-10.

### 1.2 CNN Architectures
- **ResNet-18:** Uses residual connections to avoid vanishing gradient issues, enabling deeper feature extraction.
- **MobileNet:** Employs depthwise separable convolutions, offering computational efficiency suitable for mobile environments.
- **EfficientNet:** Balances depth, width, and resolution, optimizing accuracy with minimal resources.

### 1.3 Project Goals
1. **Architecture Comparison** - Evaluate model performance on feature extraction.
2. **Hyperparameter Tuning** - Optimize learning rates and weight decay.
3. **Data Augmentation** - Enhance model robustness by expanding input diversity.

---

## 2. Method Implementation

### 2.1 Data Preprocessing and Splitting
- **Transformations:** Applied `ToTensor()` and normalized CIFAR-10 images to stabilize training.
- **Splitting:** Created an 80/20 training-validation split to balance data availability and prevent overfitting.

### 2.2 CNN Model Initialization
- **ResNet-18:** Chosen for its deep feature extraction capability.
- **MobileNet:** Selected for computational efficiency.
- **EfficientNet:** Included for its balanced architecture.

### 2.3 Training Process
- **Loss Function:** CrossEntropyLoss for multi-class classification.
- **Optimizer:** SGD with momentum to accelerate convergence.
- **Epochs:** Trained for 10 epochs, recording training and validation metrics.

### 2.4 Validation and Test Procedure
Best-performing model was saved based on validation accuracy and later evaluated on the test set for unbiased performance assessment.

---

## 3. Experimental Analysis

### 3.1 Experiment 1: Architecture Comparison
Trained ResNet-18, MobileNet, and EfficientNet for 10 epochs, comparing accuracy and stability.

| Model       | Training Accuracy | Validation Accuracy |
|-------------|--------------------|---------------------|
| ResNet-18   | 92.36%            | 68.59%             |
| MobileNet   | 66.02%            | 59.45%             |
| EfficientNet| 35.93%            | 39.27%             |

**Observations:**  
- ResNet-18’s residual connections provided stable training and generalization.
- MobileNet was efficient but limited in feature extraction.
- EfficientNet-B0 underperformed, likely due to insufficient complexity for CIFAR-10.

### 3.2 Experiment 2: Hyperparameter Tuning
Tested ResNet-18 with learning rates (0.01, 0.001) and weight decay (0, 1e-4).

| Learning Rate | Weight Decay | Validation Accuracy |
|---------------|--------------|---------------------|
| 0.01          | 0            | 67.65%             |
| 0.01          | 1e-4         | 69.35%             |
| 0.001         | 0            | 59.47%             |
| 0.001         | 1e-4         | 57.55%             |

**Results:**  
Best results achieved with learning rate 0.01 and weight decay 1e-4, balancing accuracy and preventing overfitting.

### 3.3 Experiment 3: Data Augmentation
Compared Basic, Moderate, and Aggressive augmentation strategies.

| Augmentation Level | Validation Accuracy |
|--------------------|---------------------|
| Basic              | 68.00%             |
| Moderate           | 72.20%             |
| Aggressive         | 68.58%             |

**Findings:**  
Moderate augmentation (horizontal flips) achieved the highest accuracy by enhancing diversity without excessive complexity.

---

## 4. Results Summary and Comparisons

### 4.1 Final Model Evaluation
The optimized ResNet-18 model achieved a test accuracy of 55.03%, with minimal overfitting, evidenced by steady training and validation curves.

### 4.2 Confusion Matrix Analysis
Confusion matrix shows high recall for "automobile" and "ship" classes but low recall for visually similar classes like "cat" and "dog."

### 4.3 Classification Report Analysis
| Class       | Precision | Recall | F1-Score |
|-------------|-----------|--------|----------|
| Airplane    | 0.75      | 0.52   | 0.61     |
| Automobile  | 0.79      | 0.56   | 0.66     |
| ...         | ...       | ...    | ...      |
| Ship        | 0.80      | 0.61   | 0.69     |

**Conclusion:**  
ResNet-18 achieved high performance on distinguishable classes but struggled with visually similar categories, suggesting areas for improvement.

---

## 5. Code
All implementation details are available in the [GitHub Repository](https://github.com/jithendra-varma/CNNs_for_image_classification).

---

## 6. Conclusion

### 6.1 Key Findings
ResNet-18 emerged as the best model, with residual connections facilitating deep learning of complex features. Moderate data augmentation and hyperparameter tuning further enhanced performance.

### 6.2 Reflection on Design Choices
The integrated approach of architecture selection, augmentation, and tuning maximized model efficacy, underscoring the importance of balanced data and hyperparameter optimization.

### 6.3 Future Work
Further improvements could involve:
- Testing deeper architectures like ResNet-34
- Ensemble methods for accuracy enhancement
- Advanced augmentation (cutout, mixup) for robustness

---

## References
- Goodfellow, I., Bengio, Y. & Courville, A. (2016), Deep Learning, MIT Press.
- He, K. et al. (2016), Deep residual learning for image recognition, CVPR.
- LeCun, Y. et al. (2015), Deep learning, Nature.
- Howard, A. G. et al. (2017), MobileNets: Efficient CNNs for mobile vision applications, arXiv.
- Tan, M. & Le, Q. V. (2019), EfficientNet: Rethinking model scaling for CNNs, ICML.
- Krizhevsky, A. (2009), Learning multiple layers of features from tiny images, University of Toronto.
- Rawat, W. & Wang, Z. (2017), Deep CNNs for image classification: A comprehensive review, Neural Computation.

---

