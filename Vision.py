import cv2
import numpy as np
import tensorflow as tf

#Training:
# to_categorical = tf.keras.utils.to_categorical
# Sequential = tf.keras.models.Sequential
#
# # Function to load MNIST data from binary files
# def load_mnist(image_path, label_path):
#     with open(label_path, 'rb') as lbpath:
#         magic, num = struct.unpack(">II", lbpath.read(8))
#         labels = np.fromfile(lbpath, dtype=np.uint8)
#
#     with open(image_path, 'rb') as imgpath:
#         magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
#         images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 28, 28)
#
#     return images, labels
#
# # Paths to the MNIST dataset files
# dataset_path = '/home/talnz/PythonProjects/VisionCNN/Dataset'
# train_images_path = os.path.join(dataset_path, 'train-images.idx3-ubyte')
# train_labels_path = os.path.join(dataset_path, 'train-labels.idx1-ubyte')
# test_images_path = os.path.join(dataset_path, 't10k-images.idx3-ubyte')
# test_labels_path = os.path.join(dataset_path, 't10k-labels.idx1-ubyte')
#
# # Load the data
# train_images, train_labels = load_mnist(train_images_path, train_labels_path)
# test_images, test_labels = load_mnist(test_images_path, test_labels_path)
#
# # Normalize the images
# train_images = train_images.astype('float32') / 255.0
# test_images = test_images.astype('float32') / 255.0
#
# # One-hot encode the labels
# train_labels = to_categorical(train_labels, 10)
# test_labels = to_categorical(test_labels, 10)
#
# # Build a neural network model
# model = Sequential([
#     Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
#     MaxPooling2D(pool_size=(2, 2)),
#     Conv2D(64, kernel_size=(3, 3), activation='relu'),
#     MaxPooling2D(pool_size=(2, 2)),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dropout(0.5),
#     Dense(10, activation='softmax')
# ])
#
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#
# # Reshape images to fit the model input
# train_images = train_images.reshape(-1, 28, 28, 1)
# test_images = test_images.reshape(-1, 28, 28, 1)
#
# # Train the model
# model.fit(train_images, train_labels, epochs=10, batch_size=128, validation_data=(test_images, test_labels))
#
# # Evaluate the model
# test_loss, test_acc = model.evaluate(test_images, test_labels)
# print(f'Test accuracy: {test_acc}')
#
# # Save the model
# model.save('/home/talnz/PythonProjects/VisionCNN/mnist_model.h5')
#
# # Display random sample images with labels
# def display_sample_images(images, labels, num_samples=12):
#     random_indices = np.random.choice(images.shape[0], num_samples, replace=False)
#     sample_images = images[random_indices]
#     sample_labels = labels[random_indices]
#
#     plt.figure(figsize=(10, 5))
#     for i in range(num_samples):
#         plt.subplot(2, 6, i + 1)
#         plt.imshow(sample_images[i].reshape(28, 28), cmap='gray')
#         plt.title(np.argmax(sample_labels[i]))
#         plt.axis('off')
#     plt.show()
#
# # Display sample images
# display_sample_images(train_images, train_labels)


# Load the trained model
model = tf.keras.models.load_model('/home/talnz/PythonProjects/VisionCNN/mnist_model_v2.h5')


# Function to preprocess the image for model prediction
def preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (28, 28))
    image = image / 255.0
    image = image.reshape(-1, 28, 28, 1)
    return image


# Function to predict the digit
def predict_digit(image):
    processed_image = preprocess(image)
    prediction = model.predict(processed_image)
    return np.argmax(prediction)


# Capture video from the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Assuming the ROI containing the digit is the center of the frame
    height, width, _ = frame.shape
    roi = frame[height // 2 - 50:height // 2 + 50, width // 2 - 50:width // 2 + 50]

    digit = predict_digit(roi)
    cv2.putText(frame, f'Prediction: {digit}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Draw a rectangle around the ROI
    cv2.rectangle(frame, (width // 2 - 50, height // 2 - 50), (width // 2 + 50, height // 2 + 50), (0, 255, 0), 2)

    cv2.imshow('Live Digit Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
