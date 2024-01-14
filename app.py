import streamlit as st 
from functions import *
st.title('Driver detection with image')


model = st.selectbox(label='select model',options=['resnet50','fastai_vgg'])

image = st.file_uploader(label='upload image on which you want to detect',type = ['png','jpg'])
st.toast("Working on it...")


if image is not None:
    st.image(image= image,caption = 'uploaded image')
    x = predict_with_image(image, model=model)
    st.subheader(x)
    st.balloons()
else:
    st.write('upload the image first...')
