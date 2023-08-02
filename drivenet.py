import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.utils import to_categorical



#Load the data and split it into train and test
train_data = np.load('training_data_v2.npy', allow_pickle=True)
WIDTH = 160
HEIGHT = 120

train = train_data[:-500]
test = train_data[-500:]

X_train = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,1)
Y_train = [i[1] for i in train]

X_test = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1)
Y_test = [i[1] for i in test]

X_train = np.asarray(X_train)
Y_train = np.asarray(Y_train)
X_test = np.asarray(X_test)
Y_test = np.asarray(Y_test)



# Define the CNN model
model = Sequential()
model.add(Conv2D(32, kernel_size=(5,5), activation='relu', input_shape=(WIDTH,HEIGHT,1)))
#The first layer is a convolutional layer with 32 filters, each of size (3,3)

model.add(MaxPooling2D(pool_size=(2,2)))
#The second layer is a max pooling layer with a pool size of (2,2)

model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
#The third layer is another convolutional layer with 64 filters, each of size (3,3), and ReLU activation.

model.add(MaxPooling2D(pool_size=(2,2)))
#The fourth layer is another max pooling layer with a pool size of (2,2).

model.add(Flatten())
#The fifth layer is a flatten layer, which flattens the 2D output of the previous layer into a 1D vector.

model.add(Dense(512, activation='relu'))
#The layer is a fully connected (Dense) layer with 128 neurons and ReLU activation.

model.add(Dense(128, activation='relu'))
#The layer is a fully connected (Dense) layer with 128 neurons and ReLU activation.

model.add(Dense(3, activation='sigmoid'))
#The layer is another fully connected layer with 3 neurons, one for each pressable button (W,A,D)


#model.summary()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Train the model
model.fit(X_train, Y_train, epochs=5, validation_data=(X_test, Y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, Y_test)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

# Save the model
model.save('model.h5')