import tensorflow as tf
import numpy as np
from matplotlib import pyplot
import streamlit as st

st.set_page_config(page_title = "Digit Recognition")
st.title("Digit Recognition")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
df = ((x_train, y_train), (x_test, y_test))

x_train = (x_train/255)
x_test = (x_test/255)

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(30, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

#model.summary()

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

st.write("Training started...")
history = model.fit(x_train, y_train, epochs=5)
model.save("digit_cnn.keras")
test_loss, test_accuracy = model.evaluate(x_test, y_test)
st.write("Test Accuracy:", test_accuracy)

digit_index = st.number_input(
    "Enter image number (0-9999)",
    min_value=0,
    max_value=9999,
    value=0
)
prediction = model.predict(
    x_test[digit_index].reshape(1,28,28,1))

predicted_digit = np.argmax(prediction)
st.write("Predicted Digit:", predicted_digit)
st.write("Actual Digit:", y_test[digit_index])

fig, ax = pyplot.subplots()
ax.matshow(x_train[digit_index].squeeze())
st.pyplot(fig)
