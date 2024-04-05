
import streamlit as st
import keras
import numpy as np
from fastai.vision.all import *
#from fastai.vision.widgets import *
from keras.preprocessing import image  # For image preprocessing
from keras.models import load_model  # For model loading
from keras.preprocessing.image import img_to_array, load_img
from PIL import Image  
from streamlit_lottie import st_lottie
import plotly.express as px  # For plotting



st.set_page_config(
     page_title="Spice Classifier",
     page_icon="🍌",
     initial_sidebar_state="expanded",
     layout='wide',
     menu_items={
         'Get Help': 'https://github.com/IBronko/',
         'Report a bug': "https://github.com/IBronko/fruit-image-classifier/issues",
         'About': "# This is a personal project."
     }
 )
st.sidebar.markdown("## Navigate around the app")

select_event = st.sidebar.selectbox('Menu',
                                    ['HOME', 'PREDICT', 'GRAPHS', 'ABOUT ME'])

if select_event == 'HOME':
    st.markdown("<h1 style='text-align: center; color: cream'>Welcome, I am your personal Spice classifier.</h1>", unsafe_allow_html=True)
    def load_lottiefile(filepath: str):
        with open(filepath, "r") as f:
            return json.load(f)
        
    lottie_coding = load_lottiefile("lottie/cat.json")
    col1, col2, col3 = st.columns(3)
    with col2:   
        st_lottie(
         lottie_coding,
         speed=1,
         reverse=False,
         loop=True,
         quality="medium", # medium ; high,
         key=None,
        )
    st.subheader("Hmmmmm, I'm ready to predict your spices")



elif select_event == 'PREDICT':

# Load the pre-trained model
 incep_model = load_model("my_custom_inceptionv3.h5")  # Replace with your model path

# Title and file upload section
 st.title("Spice Classification App")
 uploaded_file = st.file_uploader("Choose an image...", type="jpg")



 class_labels = ["Bayleaf", "Black Cardamom","Black pepper","Chili","Clove", "Coriander", "Cumin", "Fenugreek Seeds", "Ginger", "Green Cardamom", "Mustard seeds"]

 def predict_image(image_path):
  img = Image.open(image_path) 
  # Preprocess the image
  img_array = img_to_array(img)
  img_array = np.expand_dims(img_array, axis=0)
  img_array = img_array / 255.0
  # Make prediction
  predictions = incep_model.predict(img_array)
  # Process the predictions (e.g., get the predicted class)
  predicted_class = np.argmax(predictions[0]) 
  predicted_class_index = np.argmax(predictions[0])
  predicted_class_label = class_labels[predicted_class]
  return predicted_class_label

 if uploaded_file is not None:
  st.image(uploaded_file)  
  try:
     predicted_class = predict_image(uploaded_file)
     st.success(f"Predicted class: {predicted_class}")
  except:
     st.write("Sorry, I don't know that spice")

elif select_event == 'GRAPHS':

 epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Example epochs
 cat_accuracy_values = [0.7653, 0.8709, 0.8990, 0.9096, 0.9212, 0.9301, 0.9345, 0.9391, 0.9412, 0.9488]
 valcat_accuracy_values = [0.9070, 0.9200, 0.9250, 0.9403, 0.9370, 0.9377, 0.9417, 0.9483, 0.9410, 0.9457]
 # Prepare data for plotting
 data = {'epoch': epochs, 'Training accuracy': cat_accuracy_values, 'Validation accuracy':valcat_accuracy_values}

# Create accuracy chart
 fig = px.line(data, x='epoch', y=['Training accuracy','Validation accuracy'], title='Inception Model Accuracy', color_discrete_sequence=['orange', 'blue'])

# Display the chart in Streamlit
 st.plotly_chart(fig)

else:
   with st.expander("Info"):
     st.markdown("""
         - I have been trained by fine-tuning a __InceptionV3__ convolutional neural network
         - For each spice class, I have been provided around 1,000 images to learn from
         - After 10 training runs (epochs), this was the result on the validation set:    
     """)
    #  st.image("images/confusion_matrix.png")

   with st.expander("Dataset"):
        st.markdown("""
         - I have been trained on spice11 dataset which consists of 11 classes of spices with 1,000-1,500 images each and a total of 15,000 images
         - This dataset have been collected from various sources and varied backgrounds to improve my accuracy   
     """)
   with st.expander("Github link"):
      st.link_button("https://github.com/shhhhreya/Spice-Classification")

        