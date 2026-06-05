import tensorflow as tf
import numpy as np
from matplotlib import pyplot
import streamlit as st

st.set_page_config(page_title = "Digit Recognition")
st.title("Digit Recognition")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
df = ((x_train, y_train), (x_test, y_test))
#st.subheader ("Dataset Preview")
#st.write("X_Train", x_train.shape)
#st.write("Y_Train", y_train.shape)
#st.write("X_Test", x_test.shape)
#st.write("Y_Test", y_test.shape)

x_train = (x_train/255)
x_test = (x_test/255)

x_train_flattened = x_train.reshape(len(x_train), 784)
x_test_flattened = x_test.reshape(len(x_test), 784)
#st.write("X_Test Flattened", x_train_flattened.shape)
#st.write("Y_Test Flattened", x_test_flattened.shape)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
history = model.fit(x_train_flattened, y_train, epochs=10)

test_loss, test_accuracy = model.evaluate(x_test_flattened, y_test)
st.write("Test Accuracy:", test_accuracy)

digit_index = st.number_input(
    "Enter image number (0-9999)",
    min_value=0,
    max_value=9999,
    value=0
)
prediction = model.predict(
    x_test_flattened[digit_index].reshape(1, 784),
    verbose=0
)

predicted_digit = np.argmax(prediction)
st.write("Predicted Digit:", predicted_digit)
st.write("Actual Digit:", y_test[digit_index])

fig, ax = pyplot.subplots()
ax.matshow(x_train[digit_index])
st.pyplot(fig)
