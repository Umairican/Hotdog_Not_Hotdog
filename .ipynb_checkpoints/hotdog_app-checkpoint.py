import streamlit as st
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image
import tensorflow as tf
import numpy as np
# title
st.title('Hotdog, Not Hotdog')
# model
model = keras.models.load_model('./hnh_rcnn/')
# file uploader
# load an image to classify
uploaded_file = st.file_uploader("Choose an image:", type=["png","jpg","jpeg"])

#hotdoggy = load_img(st.file_uploader("Choose an image:", type="jpg"), target_size=(256, 256))



if uploaded_file is not None:
    hotdog = Image.open(uploaded_file)
    rgb_im = hotdog.convert('RGB')
    hotdog_arr = tf.keras.preprocessing.image.img_to_array(rgb_im) / 255
    resized_hotdog = tf.image.resize(hotdog_arr, (256, 256))
    hotdog_array = np.array(resized_hotdog).reshape(1,256,256,3)


#These are for displaying to our intrepid customer
    st.write("")
    st.write("Classifying...")
    # model prediction
    label = model.predict(hotdog_array)
    # write results
    st.write("You are this likely to be in hotdogland")
    st.write(label)
