import tensorflow as tf
import os
from tensorflow.keras.applications import EfficientNetB3, Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Average
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
import numpy as np

# Dataset paths
train_dir = r"C:\Users\eliza\Downloads\Dataset\Train"
validation_dir = r"C:\Users\eliza\Downloads\Dataset\Validation"
test_dir = r"C:\Users\eliza\Downloads\Dataset\Test"
cls_pic_dir = r"C:\Users\eliza\Downloads\Cls_pic"

# Constants
IMG_SIZE = (224, 224)  # Resize images to this size
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
EPOCHS = 5

# Image preprocessing function
def preprocess_image(image_path, label):
    try:
        img = tf.io.read_file(image_path)
        
        # Decode based on the file extension
        img = tf.image.decode_image(img, channels=3)
        
        # Resize and normalize the image
        img = tf.image.resize(img, IMG_SIZE)
        img = img / 255.0
        return img, label
    except Exception as e:
        tf.print(f"Error processing {image_path}: {e}")
        return None, label  # Skip this image


# Function to load datasets
def load_dataset(directory, label_map):
    files = []
    labels = []
    for label_name, label_value in label_map.items():
        folder_path = os.path.join(directory, label_name)
        if os.path.exists(folder_path):
            file_paths = [
                os.path.join(folder_path, fname) 
                for fname in os.listdir(folder_path) 
                if fname.endswith(('.jpg', '.png'))
            ]
            files.extend(file_paths)
            labels.extend([label_value] * len(file_paths))
    return tf.data.Dataset.from_tensor_slices((files, labels))

# Prepare datasets
label_map = {'real': 0, 'fake': 1}

train_dataset = load_dataset(train_dir, label_map)
val_dataset = load_dataset(validation_dir, label_map)
test_dataset = load_dataset(test_dir, label_map)
cls_pic_dataset = load_dataset(cls_pic_dir, label_map)

# Combine cls_pic dataset with train dataset
train_dataset = train_dataset.concatenate(cls_pic_dataset)

# Preprocess datasets
train_dataset = (train_dataset
                 .map(preprocess_image, num_parallel_calls=AUTOTUNE)
                 .shuffle(1000)
                 .batch(BATCH_SIZE)
                 .prefetch(AUTOTUNE))

val_dataset = (val_dataset
               .map(preprocess_image, num_parallel_calls=AUTOTUNE)
               .batch(BATCH_SIZE)
               .prefetch(AUTOTUNE))

test_dataset = (test_dataset
                .map(preprocess_image, num_parallel_calls=AUTOTUNE)
                .batch(BATCH_SIZE)
                .prefetch(AUTOTUNE))

# Base models
base_model_eff = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model_xcp = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Custom layers for both models
x1 = GlobalAveragePooling2D()(base_model_eff.output)
x1 = Dense(256, activation='relu')(x1)
x2 = GlobalAveragePooling2D()(base_model_xcp.output)
x2 = Dense(256, activation='relu')(x2)

# Combine the models
combined = Average()([x1, x2])
predictions = Dense(1, activation='sigmoid')(combined)

# Ensemble model
model = Model(inputs=[base_model_eff.input, base_model_xcp.input], outputs=predictions)

# Freeze base model layers
for layer in base_model_eff.layers:
    layer.trainable = False
for layer in base_model_xcp.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
print("Training the ensemble model with frozen layers...")
model.fit(
    [train_dataset, train_dataset],  # Pass both models' inputs as a list
    validation_data=([val_dataset, val_dataset], val_dataset),
    epochs=EPOCHS
)

# Fine-tune deeper layers
for layer in base_model_eff.layers[-30:]:
    layer.trainable = True
for layer in base_model_xcp.layers[-30:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

print("Fine-tuning the deeper ensemble layers...")
model.fit(
    [train_dataset, train_dataset],  # Pass both models' inputs as a list
    validation_data=([val_dataset, val_dataset], val_dataset),
    epochs=EPOCHS
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate([test_dataset, test_dataset], test_dataset)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Prediction function
def predict_image(model, img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    prediction = model.predict([img_array, img_array])[0][0]
    return "real" if prediction > 0.5 else "fake"

# Example usage
example_images = [
    r"C:\Users\eliza\Downloads\test_gen1.png",
    r"C:\Users\eliza\Downloads\test_gen2.png",
    r"C:\Users\eliza\Downloads\test_real1.png",
    r"C:\Users\eliza\Downloads\test_real2.png",
    r"C:\Users\eliza\Downloads\real_4.png"
]

for img_path in example_images:
    result = predict_image(model, img_path)
    print(f"Prediction for {img_path}: {result}")
