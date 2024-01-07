import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras import layers,models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D 
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

############################################################################################################

train_df = pd.read_csv('/kaggle/input/cassava-leaf-disease-classification/train.csv')
train_df.head()

############################################################################################################

train_df['label'].value_counts()

############################################################################################################

from sklearn.metrics import accuracy_score
y_pred = [3] * len(train_df.label)
print("The baseline accuracy is {}".format(accuracy_score(y_pred, train_df.label)))

############################################################################################################

import os
Dir = '/kaggle/input/cassava-leaf-disease-classification'
CFG = {
    'train_folder' : 'train_images',
    'test_folder' : 'test_images',
}

############################################################################################################

Batch_size = 32

############################################################################################################

train_df['label'] = train_df['label'].astype('str')
datagen = ImageDataGenerator(
    horizontal_flip = True,
    vertical_flip = True,
    validation_split = 0.2,
    rotation_range=0.5,  # Randomly rotate images in the range
    zoom_range = 0.2, # Randomly zoom image
    width_shift_range=0.1,  # Randomly shift images horizontally
    height_shift_range=0.1,  # Randomly shift images vertically
)

train_datagen = datagen.flow_from_dataframe(
    train_df,
    directory = os.path.join(Dir, "train_images"),
    batch_size = Batch_size,
    target_size = (300, 300),
    subset = "training",
    seed = 42,
    x_col = "image_id",
    y_col = "label",
    class_mode = "categorical"
)

############################################################################################################

val_gen = ImageDataGenerator(
    validation_split = 0.2,
)

val_datagen = val_gen.flow_from_dataframe(
    train_df,
    directory = os.path.join(Dir, "train_images"),
    batch_size = Batch_size,
    target_size = (300,300),
    subset = "validation",
    seed = 42,
    x_col = "image_id",
    y_col = "label",
    class_mode = "categorical"
)

############################################################################################################

img, label = next(train_datagen)

############################################################################################################

dir_efficientnetb3 = '/kaggle/input/keras-efficientnetb3-no-top-weights/efficientnetb3_notop.h5

############################################################################################################

from tensorflow.keras.applications.efficientnet import EfficientNetB3

############################################################################################################

model = Sequential()
model.add(EfficientNetB3(include_top = False, weights = dir_efficientnetb3,
                        input_shape = (300, 300, 3), drop_connect_rate = 0.3))
model.add(GlobalAveragePooling2D())
model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.3))
model.add(Dense(5, activation = 'softmax'))

loss_set = CategoricalCrossentropy(label_smoothing = 0.0001, 
                               name = 'categorical_crossentropy')

model.compile(optimizer = Adam(), 
              loss = loss_set, 
              metrics = ["categorical_accuracy"])
model.layers[0].trainable = False
model.summary()

############################################################################################################

tf.keras.utils.plot_model(model)

############################################################################################################

early_stop = EarlyStopping(monitor = 'val_loss', patience = 10)

history = model.fit_generator(
    train_datagen,
    epochs = 3,
    verbose = 1,
    validation_data = val_datagen,
    callbacks = [early_stop]
)

############################################################################################################

import matplotlib.pyplot as plt

# Access the accuracy values from the history object
train_accuracy = history.history['categorical_accuracy']
val_accuracy = history.history['val_categorical_accuracy']

# Access the epochs
epochs = range(1, len(train_accuracy) + 1)

# Plotting the accuracy
plt.figure(figsize=(10, 6))
plt.plot(epochs ,train_accuracy, 'g', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

print(history.history['categorical_accuracy'])
print(history.history['val_categorical_accuracy'])

############################################################################################################

from PIL import Image
import pandas as pd
import numpy as np
from numpy import random
import os
import tensorflow as tf
from tensorflow.keras import layers
from tqdm import tqdm
crop_size=224

############################################################################################################

def scan_over_image(img_path, crop_size=300):
    img = Image.open(img_path)
    img_height,img_width = img.size
    
    img = np.array(img)
    print("img shape :", img.shape)
    
    print("height :",img_height, 'weight :',img_width)
    
    x_img_origins = [0,img_width-crop_size]
    y_img_origins = [0,img_height-crop_size]
    
    print("x_img_origins : ",x_img_origins, "y_img_origins",y_img_origins)
    
    img_list = []
    for x in x_img_origins:
        for y in y_img_origins:
            img_list.append(img[x:x+crop_size , y:y+crop_size,:])
            print("x =", x, " y =", y, " x+crop_size =", x+crop_size," y+crop_size =", y+crop_size, )
    return np.array(img_list)
    
############################################################################################################
    
test_time_augmentation_layers = tf.keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        layers.experimental.preprocessing.RandomZoom((-0.2, 0)),
        layers.experimental.preprocessing.RandomContrast((0.2,0.2))
    ]
)

############################################################################################################

def predict_and_vote(image_filename, folder, TTA_runs=4) :
    localised_predictions = []
    local_image_list = scan_over_image(folder+image_filename)
    for local_image in local_image_list:
        duplicated_local_image = tf.convert_to_tensor(np.array([local_image for i in range(TTA_runs)]))
        augmented_images = test_time_augmentation_layers(duplicated_local_image)
        predictions = model.predict(augmented_images)
        localised_predictions.append(np.sum(predictions, axis=0))
    global_predictions = np.sum(np.array(localised_predictions),axis=0)
    final_prediction = np.argmax(global_predictions)
    
    return final_prediction
    
############################################################################################################
    
def run_predictions(image_list, folder):
    predictions = [] 
    with tqdm(total=len(image_list)) as pbar:
        for image_filename in image_list:
            predictions.append(predict_and_vote(image_filename, folder))
            pbar.update(1)
    return predictions
    
############################################################################################################    
    
sample_df = pd.read_csv('/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv')

############################################################################################################

TEST_IMAGE_PATH = '../input/cassava-leaf-disease-classification/test_images/'
SUBMISSION_PATH = 'submission.csv'

############################################################################################################

submission_df = pd.DataFrame(columns=["image_id","label"])
submission_df["image_id"] =  os.listdir(os.path.join(Dir, "test_images") + '/')
submission_df["label"] = run_predictions(submission_df["image_id"], os.path.join(Dir, "test_images") + '/')
submission_df.to_csv(SUBMISSION_PATH, index=False)
print(submission_df)

############################################################################################################

import cv2

img = Image.open('/kaggle/input/cassava-leaf-disease-classification/test_images/2216849948.jpg')
img = np.array(img)

# Resize the image to (224, 224, 3)
img = cv2.resize(img, (300, 300))

# Add an extra dimension to represent the batch size
img = np.expand_dims(img, axis=0)

preds = model.predict(img)

print(np.argmax(preds))

############################################################################################################

from sklearn.metrics import roc_curve, auc
from tensorflow.keras.utils import to_categorical

val_true_labels = np.array(val_datagen.classes)
preds_test = model.predict(val_datagen)

loss, accuracy = model.evaluate(val_datagen, verbose = 1)

print("accuracy :", accuracy)

class_preds_test =  np.argmax(preds_test, axis = 1)

num_pre = (val_true_labels != class_preds_test).astype(int)

num_pre = np.array(num_pre)
# zero_array = np.array(zero_array)

print(val_true_labels)
print(class_preds_test)

print(np.sum(val_true_labels == class_preds_test))
print(np.sum(val_true_labels != class_preds_test))