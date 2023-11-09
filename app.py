import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import altair as alt  # Import Altair
from streamlit_option_menu import option_menu

def page_home():
    st.title("Cassava Leaf Disease Classification")
    st.write("The importance of cassava as an important food crop. Whether it is processed into instant noodles, monosodium glutamate, used as animal feed, and much more, from its ability to be used in a variety of applications.")
    st.write("Including the ability to withstand unfavorable environmental conditions, many areas with unsuitable environmental conditions for cultivation often plant cassava to be used as the main source of food.")
    st.write("However, at present, production from cultivation is not at full efficiency due to virus infection of cassava. There are four types of viruses that are the main problem affecting the quality of cassava:")
    st.write("1. Cassava Bacterial Blight (CBB)")
    st.write("2. Cassava Brown Streak Disease (CBSD)")
    st.write("3. Cassava Green Mottle (CGM)")
    st.write("4. Cassava Mosaic Disease (CMD)")
    st.write("All 4 types of viruses can be observed and differentiated by looking at abnormal leaf characteristics.")
    st.write("From the above sources, we can recognize that we can identify infected cassava by looking at the appearance of the leaves.")
    st.write("Therefore, if we use Machine Learning to help with this task, it will help reduce labor costs and time costs spent sorting cassava.")

def page_prediction():
    st.title("Image Classava Prediction App")

    with open('style.css') as f :
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        
    model = tf.keras.models.load_model('cassava_model.h5')
        
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    class_name = ["Cassava Bacterial Blight (CBB)", 
                "Cassava Brown Streak Disease (CBSD)",
                "Cassava Green Mottle (CGM)",
                "Cassava Mosaic Disease (CMD)",
                "Healthy"
    ]
    class_name_cut = ["CBB", 
                "CBSD",
                "CGM",
                "CMD",
                "Healthy"
    ]



    # If the user has uploaded a file
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = image.resize((300, 300))
        image_array = np.array(image)
        image_array = np.expand_dims(image_array, axis=0)
        prediction = model.predict(image_array)
        class_prediction = np.argmax(prediction, axis=1)
        st.image(image, caption="Cassava Image", use_column_width=True)
        st.subheader(f"Predicted class : {class_name[class_prediction[0]]} ({class_prediction[0]})")

        col1, col2, col3= st.columns(3)
        col1.metric("CCB", "{:.2f} %".format(prediction[0][0]*100))
        col2.metric("CBSD", "{:.2f} %".format(prediction[0][1]*100))
        col3.metric("CGM", "{:.2f} %".format(prediction[0][2]*100))
        col4, col5 = st.columns(2)
        col4.metric("CMD", "{:.2f} %".format(prediction[0][3]*100))
        col5.metric("Healthy", "{:.2f} %".format(prediction[0][4]*100))
        
        # Create a beautiful and modern-looking bar plot using Altair
        st.subheader("Class Prediction Percentages")
        data = pd.DataFrame({'Class': class_name_cut, 'Percentage': prediction[0] * 100})
        chart = alt.Chart(data).mark_bar().encode(
            x=alt.X('Class', sort=None),
            y='Percentage',
            color='Class'
        ).properties(width=500, height=300)
        st.altair_chart(chart, use_container_width=True)
        
        st.write('Number of Cassava Leaf Disease')
        st.write(class_name)




def page_makeby():
    st.title("Make by")
    st.write("Nattachai Aroonkijjarurn 6404062610073 S1")
    st.write("Surachet Poonsinpokasup 6404062630589 S1")
    
    st.write()
    st.subheader('Advisor')
    st.write('Asst.Prof.Luepol Pipanmekaporn')

def reference_page() :
    st.title('reference')
    st.write('1. https://supina-p.medium.com/%E0%B9%80%E0%B8%97%E0%B8%84%E0%B8%99%E0%B8%B4%E0%B8%84-data-augmentation-%E0%B9%81%E0%B8%A5%E0%B8%B0-batch-normalization-%E0%B8%94%E0%B9%89%E0%B8%A7%E0%B8%A2%E0%B8%81%E0%B8%B2%E0%B8%A3%E0%B8%97%E0%B8%B3-regularization-%E0%B9%81%E0%B8%9A%E0%B8%9A%E0%B8%AA%E0%B8%A1%E0%B8%B1%E0%B8%A2%E0%B9%83%E0%B8%AB%E0%B8%A1%E0%B9%88-5eadee732569')
    st.write('2. https://www.kaggle.com/competitions/cassava-leaf-disease-classification.')


# Create the option menu
selected_page = option_menu(
    None,
    ["Home", "Prediction", "make by", "reference"],
    default_index=0,
    icons=["house", "image", "envelope", "file-earmark-text"],
)



# Render the selected page
if selected_page == "Home":
    page_home()
elif selected_page == "Prediction":
    page_prediction()
elif selected_page == "make by":
    page_makeby()
elif selected_page == "reference" :
    reference_page()