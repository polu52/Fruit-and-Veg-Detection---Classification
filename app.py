import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2

model=load_model('fruit_veg_model2.h5')

def process_image(img):
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (170, 170))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

st.title("Fruit & Veg Detection ")
st.write('Upload an image and model predict which fruit or veg is this')

file=st.file_uploader('Upload an image',type=['jpg','jpeg','png'])


if file is not None:
    img=Image.open(file)
    st.image(img,caption='Uploaded image')
    image=process_image(img)
    prediction=model.predict(image)
    predicted_class=np.argmax(prediction)

    class_names={0:'apple',
                1:'banana',
                2:'beetroot',
                3:'bell pepper',
                4:'cabbage',
                5:'capsicum',
                6:'carrot',
                7:'cauliflower',
                8:'chilli pepper',
                9:'corn',
                10:'cucumber',
                11:'eggplant',
                12:'garlic',
                13:'ginger',
                14:'grapes',
                15:'jalepeno',
                16:'kiwi',
                17:'lemon',
                18:'lettuce',
                19:'mango',
                20:'onion',
                21:'orange',
                22:'paprika',
                23:'pear',
                24:'peas',
                25:'pineapple',
                26:'pomegranate',
                27:'potato',
                28:'raddish',
                29:'soy beans',
                30:'spinach',
                31:'sweetcorn',
                32:'sweetpotato',
                33:'tomato',
                34:'turnip',
                35:'watermelon'
                }

    st.write(class_names[predicted_class])
