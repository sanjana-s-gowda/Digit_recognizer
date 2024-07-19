import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

# Load the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape the data
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Normalize the data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=12,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2
)
datagen.fit(x_train)

# Model parameters
batch_size = 128
num_classes = 10
epochs = 50  # Increased the number of epochs to 50

# Define the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))  # Reduced dropout rate

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))  # Reduced dropout rate

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))  # Adjusted dropout rate
model.add(Dense(num_classes, activation='softmax'))

# Compile the model with a lower learning rate
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=Adam(learning_rate=0.001),  # Lowered learning rate
              metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
hist = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
                 steps_per_epoch=len(x_train) // batch_size,
                 epochs=epochs,
                 verbose=1,
                 validation_data=(x_test, y_test))

print("The model has successfully trained")

# Evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Save the model
model.save('mnist_improved_v2.h5')
print("Saving the model as mnist_improved_v2.h5")
