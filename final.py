
import streamlit as st
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
     page_icon="🌶️",
     initial_sidebar_state="expanded",
     layout='wide',
     menu_items={
         'Get Help': 'https://github.com/IBronko/',
         'Report a bug': "https://github.com/IBronko/fruit-image-classifier/issues",
         'About': "# This is a personal project."
     }
 )
st.sidebar.markdown("## Navigate around the app")

select_event = st.sidebar.selectbox('',
                                    ['HOME', 'PREDICT', 'ABOUT ME'])

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
    st.success(f"This looks like: __{predicted_class}__")
    if predicted_class == 'Coriander':
        st.write("Coriander (Coriandrum sativum) is a small, bushy herb with thin stems, many branches, and umbels. The plant is native to southern Europe, but is now cultivated in many places around the world. The dried seeds and fruit of the coriander plant are used as a spice and are also known as coriander. The leaves of the coriander plant are called cilantro")
        st.link_button("Know more", "https://en.wikipedia.org/wiki/Coriander")
    
    elif predicted_class == 'Bayleaf':
        st.write("A popular spice used in pickling and marinating and to flavour stews, stuffings, and fish, bay leaves are delicately fragrant but have a bitter taste. They contain approximately 2 percent essential oil, the principal component of which is cineole.")
        st.link_button("Know more", "https://en.wikipedia.org/wiki/Bay_leaf")

    elif predicted_class == 'Black Cardamom':
        st.write("Black cardamom is a spice with a strong aroma and a smoky, camphor flavor that pairs well with savory or sweet dishes. It goes by many names, including Nepal cardamom, big cardamom (most popular in Bhutan), Indian cardamom, and greater cardamom")
        st.link_button("Know more", "https://en.wikipedia.org/wiki/Black_cardamom")

    elif predicted_class == 'Black pepper':
        st.write("Black pepper is one of the most common spices used around the world, especially in European cuisine. It's the world's most traded spice. The pepper gets its heat from a chemical called piperine, unlike other peppers that get it from capsaicin. Black pepper is sharp and still leaves the mouth feeling a little prickly.")
        st.link_button("Know more", "https://en.wikipedia.org/wiki/Black_pepper")

    elif predicted_class == 'Chili':
        st.write("Chili peppers are widely used in many cuisines as a spice to add heat to dishes. Capsaicin and related compounds known as capsaicinoids are the substances that give chili peppers their intensity when ingested or applied topically.")
        st.link_button("Know more", "https://en.wikipedia.org/wiki/Chili_pepper#:~:text=Chili%20peppers%20are%20widely%20used,when%20ingested%20or%20applied%20topically.")

    elif predicted_class == 'Clove':
        st.write("Cloves are a spice that come from the dried flower buds of the clove tree. The buds are harvested while still immature and then dried, and whole cloves are about 1 centimeter long and have a reddish-brown color with a bulbous top. Cloves are a pungent spice with an intense flavor and aroma")
        st.link_button("Know more", "https://en.wikipedia.org/wiki/Clove")
    
    elif predicted_class == 'Cumin':
        st.write("Cumin has been used in cooking since ancient times. Whole cumin seeds have a more potent flavor and should be added early in the recipe. Ground cumin is more commonly used and is a staple in most curry powders and many spice blends. Cumin works particularly well with chili flakes, as they bolster the natural spicy flavor and add a rich, earthier tone")
        st.link_button("Know more", "https://en.wikipedia.org/wiki/Cumin")
    
    elif predicted_class == 'Fenugreek Seeds':
        st.write("Fenugreek seeds are used as an ingredient in spice blends and as a flavoring agent in foods, beverages, and tobacco. They are a principle constituent of curry powder, and are also used in pickles, vegetable dishes, dal, and spice mixes such as panch phoron and sambar powder. Fenugreek seeds are often roasted to reduce inherent bitterness and to enhance flavor")
        st.link_button("Know more", "https://en.wikipedia.org/wiki/Fenugreek")

    elif predicted_class == 'Ginger':
        st.write("Ginger adds a fragrant zest to both sweet and savory foods. The pleasantly spicy “kick” from the root of Zingiber officinale, the ginger plant, is what makes ginger ale, ginger tea, candies and many Asian dishes so appealing.")
        st.link_button("Know more", "https://en.wikipedia.org/wiki/Ginger")
    
    elif predicted_class == 'Green Cardamom':
        st.write("Green cardamom, also known as true cardamom, is a spice that comes from the Elettaria cardamomum plant, which is native to southern India. The plant is related to ginger and is known for its aromatic, sweet, and slightly camphorous flavor. Green cardamom is used in both sweet and savory dishes, and is often found in masala chai (spiced tea) and traditional Indian sweets. It is also used as a garnish in dishes like basmati rice.")
        st.link_button("Know more", "https://en.wikipedia.org/wiki/Elettaria_cardamomum#:~:text=Elettaria%20cardamomum%2C%20commonly%20known%20as,sharp%2C%20strong%2C%20punchy%20aroma.")
    
    elif predicted_class == 'Mustard seeds':
        st.write("Mustard seeds are the small round seeds of three different plants: black mustard, brown Indian mustard, and white mustard. Mustards have been used in traditional folk medicine as a stimulant, diuretic, and purgative to treat a variety of ailments including peritonitis and neuralgia.")
        st.link_button("Know more", "https://en.wikipedia.org/wiki/Mustard_seed")
    
  except:
    st.warning("Sorry, I don't know that spice. Please try again with another image.")
    
 
else:
    with st.expander("Info"):
     st.markdown("""
         - I have been trained by fine-tuning a __XceptionV3__ convolutional neural network.
         - For each spice class, I have been provided around 1,000 images to learn from.
     """)
    #  st.image("images/confusion_matrix.png")

    with st.expander("Accuracy Graph"):
      epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Example epochs
      cat_accuracy_values = [0.7639, 0.9366, 0.9634, 0.9720, 0.9761, 0.9765, 0.9818, 0.9847, 0.9816, 0.9869]
      valcat_accuracy_values = [0.9427, 0.9603, 0.9590, 0.9663, 0.9643, 0.9667, 0.9680, 0.9637, 0.9683, 0.9707]
      # Prepare data for plotting
      data = {'epoch': epochs, 'Training accuracy': cat_accuracy_values, 'Validation accuracy':valcat_accuracy_values}

# Create accuracy chart
      fig = px.line(data, x='epoch', y=['Training accuracy','Validation accuracy'], title='Model Accuracy', color_discrete_sequence=['#05ab18', 'blue'])

# Display the chart in Streamlit
      st.plotly_chart(fig)

    with st.expander("Dataset"):
        st.markdown("""
         - I have been trained on spice11 dataset which consists of 11 classes of spices with 1,000-1,500 images each and a total of 15,000 images.
         - This dataset have been collected from various sources and varied backgrounds to improve my accuracy.
     """)
    with st.expander("Links"):
      #st.link_button("https://github.com/shhhhreya/Spice-Classification")
      st.link_button("Repo", "https://github.com/shhhhreya/Spice-Classification", use_container_width=True) 
      st.link_button("Dataset", "https://github.com/shhhhreya/Spice-Classification", use_container_width=True) 


        
