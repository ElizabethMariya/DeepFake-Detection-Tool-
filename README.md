# DeepFake-Detection-Tool
Developed an AI-driven detection system using Xception and EfficientNet for identifying manipulated images. Applied NLP  and statistical modeling for classification, achieving 88% accuracy and ensuring robustness across test datasets. 



 Project Overview

DeepFake manipulation is increasing across social media, creating challenges for authenticity verification.
This project detects manipulated media using a dual-backbone ensemble model that leverages the strengths of both EfficientNetB3 and Xception.

- Key Features

Dual-model ensemble (EfficientNetB3 + Xception)

Image preprocessing using TF Data Pipelines

Fine-tuning support for deeper layers

Handles mixed datasets (train/validation/test/extra "cls_pic")

Custom prediction function for real-time image inference

Achieved 88% accuracy in testing

 - Model Architecture
1. EfficientNetB3 Backbone

Lightweight & scalable

Good for high-level image representations

2. Xception Backbone

Excels at detecting fine-grained pixel distortions

Strong for DeepFake artifact detection

3. Ensemble Strategy

Global Average Pooling + Dense(256) applied separately

Combined using tf.keras.layers.Average()

Final output: Sigmoid layer for binary classification (real vs fake)

