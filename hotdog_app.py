import streamlit as st
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image
import tensorflow as tf
import numpy as np
# title
st.title('Hotdog, Not Hotdog')
image = Image.open("./assets/jian_yang_hbo_silicon_valley.0.jpeg")
st.image(image)
# model
model = keras.models.load_model('./assets/model')
# file uploader
# load an image to classify
uploaded_file = st.file_uploader("Upload a scrumptious image:", type=["png","jpg","jpeg"])

#hotdoggy = load_img(st.file_uploader("Choose an image:", type="jpg"), target_size=(256, 256))



if uploaded_file is not None:
    hotdog = Image.open(uploaded_file)
    rgb_im = hotdog.convert('RGB')
    hotdog_arr = tf.keras.preprocessing.image.img_to_array(rgb_im) / 255
    resized_hotdog = tf.image.resize(hotdog_arr, (256, 256))
    hotdog_array = np.array(resized_hotdog).reshape(1,256,256,3)


#These are for displaying to our intrepid customers
    st.write("")
    st.write("Classifying... Please grab a hotdog while you wait")
    # model prediction
    label = model.predict(hotdog_array)
    label = str(label * 100) + '%'
    # write results
    st.write("You are this likely to be in hotdogland")
    st.write(label)
    st.markdown("![Alt Text](https://media.giphy.com/media/3o7TKO3AC2o5cOkZfG/giphy.gif)")
