# Development of Webpage
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageOps
import cv2
import numpy as np
import tensorflow as tf
from keras import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.models import * 

html_temp = """ 
  <div style="background-color:pink ;padding:10px">
  <h2 style="color:white;text-align:center;"> LUNG CT IMAGE CLASSIFICATION ⚕️⚕️⚕️</h2>
  </div>
  """ 
st.markdown(html_temp, unsafe_allow_html=True) 
html_temp = """ 
  <div style="background-color:teal ;padding:10px">
  <h2 style="color:white;text-align:center;"> Done by Joshita and Devi</h2>
  </div>
  """ 
st.markdown(html_temp, unsafe_allow_html=True) 
st.header('Types of cancer covered in the dataset are:')
st.subheader('1. Adenocarcinoma')
st.write('Cancer is a medical condition which is due to abnormal, uncontrollable, uncoordinated division of cells.')
st.write('Adenocarcinoma of the lung: Lung adenocarcinoma is the most common form of lung cancer accounting for 30 percent of all cases overall and about 40 percent of all non-small cell lung cancer occurrences. Adenocarcinomas are found in several common cancers, including breast, prostate and colorectal. Adenocarcinomas of the lung are found in the outer region of the lung in glands that secrete mucus and help us breathe. Symptoms include coughing, hoarseness, weight loss and weakness.')
st.subheader('2. Large cell carcinoma')
st.write('Large-cell undifferentiated carcinoma: Large-cell undifferentiated carcinoma lung cancer grows and spreads quickly and can be found anywhere in the lung. This type of lung cancer usually accounts for 10 to 15 percent of all cases of NSCLC. Large-cell undifferentiated carcinoma tends to grow and spread quickly.')
st.subheader('3. Squamous cell carcinoma')
st.write('Squamous cell: This type of lung cancer is found centrally in the lung, where the larger bronchi join the trachea to the lung, or in one of the main airway branches. Squamous cell lung cancer is responsible for about 30 percent of all non-small cell lung cancers, and is generally linked to smoking.')
st.subheader('4. Normal')
st.write('This is normal lung CT image.')
def load():
    base_model = tf.keras.applications.EfficientNetB3(weights='imagenet', input_shape=(224,224,3), include_top=False)

    for layer in base_model.layers:
        layer.trainable=True
    model = Sequential()
    model.add(base_model)
    model.add(GaussianNoise(0.25))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1024,activation='relu'))
    model.add(BatchNormalization())
    model.add(GaussianNoise(0.25))
    model.add(Dropout(0.25))
    model.add(Dense(4, activation='sigmoid'))
    model.load_weights('CancerModel.h5')
    return model
file = st.file_uploader("Please upload any image from the local machine in case of computer or upload camera image in case of mobile.", type=["jpg", "png","jpeg"])
st.set_option('deprecation.showfileUploaderEncoding', False)
def import_and_predict(image_data, model):   
        size = (224,224)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img[np.newaxis,...]
        prediction = model.predict(img_reshape)
        return prediction
if file is None:
     st.text("Please upload an image file within the allotted file size")
else:
     image = Image.open(file)
     image = image.convert('RGB') # Convert to RGB color space
     image = image.resize((224, 224))
     model = load()
     image = np.array(image)
     image = np.expand_dims(image,axis=0) 
     print(image.shape,"<======================")
     predictions = model.predict(image)

     st.title("Detected Disease Class is ")
     classes=["Adenocarcinoma","Large cell carcinoma","Squamous cell carcinoma","Normal lung"]
     print(predictions)
     a=np.argmax(predictions,-1)
     if a==0:
       st.error('The subject under observation is suspected to have adenocarcinoma. Please ensure that you consult with a professional before pursuing any kind of treatment.')
       st.warning('the model is only 85% accurate. This is the beta version of the model. Futher enhancements has to made to get the best results.')
     elif a==1:
       st.error('The subject under observation is suspected to have large cell carcinoma. Please ensure that you consult with a professional or confirm with the other modalities present in the tool.') 
       st.warning('the model is only 85% accurate. This is the beta version of the model. Futher enhancements has to made to get the best results.')
     elif a==2:
       st.success('The subject under consideration is void of any diseases or lung/breast cancer.')
       st.warning('the model is only 85% accurate. This is the beta version of the model. Futher enhancements has to made to get the best results.')      
     else:
       st.error('The subject under observation is suspected to have squamous cell carcinoma. Please ensure that you consult with a professional or confirm with the other modalities present in the tool.')
       st.warning('the model is only 85% accurate. This is the beta version of the model. Futher enhancements has to made to get the best results.')        
