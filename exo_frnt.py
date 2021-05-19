
import pandas as pd
import streamlit as st
import numpy as np
from streamlit import caching
from keras.models import model_from_json
import base64
from keras import backend as K
import matplotlib.pyplot as plt
from scipy.ndimage.filters import uniform_filter1d, gaussian_filter

st.set_page_config(page_title ="Exoplanet Hunters",
                    initial_sidebar_state="collapsed",
                    page_icon="🔮")
tabs = ["Home","Transit method","Predict using lstm","Predict using CNN","About"]
page = st.sidebar.radio("Navigation",tabs)

if page =="Home":
    main_bg = "preblk.jpg"
    main_bg_ext = "jpg"

    st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
    }}

    </style>
    """,
    unsafe_allow_html=True
    )
    logo = 'exologo.jpeg'
    st.image(logo)
    html_temp = """ 
        <div style ="background-color:black;padding:13px"> 
        <h1 style ="color:navy;text-align:center;">Exoplanet Detection app</h1> 
        </div> 
        """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.markdown("""**"Welcome to exoplanet hunting ! Let's hunt for some planets together with Artificial Intelligence ;-)"**""")
if page =="Transit method":
    st.header("Detection techniques")
    st.subheader("Transit Method")
    transit = 'transit1.jpeg'
    st.image(transit)
    st.write("Transit photometry is currently the most effective and sensitive method for detecting extrasolar planets. It is a particularly advantageous method for space-based observatories that can stare continuously at stars for weeks or months.")
if page =="Predict using lstm":
    data = st.sidebar.file_uploader(label="Enter the data to be tested", type=['csv'])
#with st.sidebar:
 #   if st.button(label='Clear cache'):
  #      caching.clear_cache()
            
    if data is not None:
        test = pd.read_csv(data)
        x_test = test.drop('LABEL', axis=1)
        rows = st.slider("Select the input row",1,570,1)
        rows1 = rows+1
        st.write(rows)
        plot = pd.DataFrame(x_test[rows:rows1].values).T
            
        if st.button('View the flux'):
                st.title('Light curve for star {}'.format(rows))
                st.write(x_test[rows:rows1].T)
                st.line_chart(plot)
        if st.button('Predict using lstm'):
            
                
            json_file = open('LSTM_model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights("LSTM_model.h5")
            loaded_model.summary()
            y_pred = loaded_model.predict((x_test.values.reshape(x_test.shape[0], x_test.shape[1], -1))[rows:rows1])
        
            html_temp = """ 
                <div style ="background-color:black;padding:13px"> 
                <h2 style ="color:white;text-align:center;">The predicted value is </h2> 
                </div> 
                """
            st.markdown(html_temp, unsafe_allow_html=True)
            st.write(y_pred)
            st.write("hi")
            if y_pred < 0.7 :
                st.write("hi1")
                html_temp = """ 
                <div style ="background-color:pink;padding:10px"> 
                <h3 style ="color:black;text-align:center;">Nope ! </h3> 
                </div> 
                """
                st.markdown(html_temp, unsafe_allow_html=True)
            else:
                st.write("hi2")   
                html_temp = """ 
                <div style ="background-color:pink;padding:10px"> 
                <h3 style ="color:black;text-align:center;">Wohoo ! We found an exoplanet for the star !</h3> 
                </div> 
                """
                st.markdown(html_temp, unsafe_allow_html=True)
if page =="Predict using CNN":
    
        data1 = st.sidebar.file_uploader(label="Enter the data to be tested", type=['csv'])
            
        if data1 is not None:
            test = pd.read_csv(data1)
            st.write("hi")
            rows = st.slider("Select the input row",1,570,1)
            rows1 = rows+1
            st.write(rows)
            #plot = pd.DataFrame(x_test[rows:rows1].values).T
        
            x_test = test.drop('LABEL', axis=1)
            x_test = np.array(x_test)
            x_test = np.append(x_test, np.flip(x_test[0:5,:], axis=-1), axis=0)
            def detrender_normalizer(light_flux):
                flux1 = light_flux
                flux2 = gaussian_filter(flux1, sigma=10)
                flux3 = flux1 - flux2
                flux3normalized = (flux3-np.mean(flux3)) / (np.max(flux3)-np.min(flux3))
                return flux3normalized

            x_test_p = detrender_normalizer(x_test)
            x_test = ((x_test - np.mean(x_test, axis=1).reshape(-1,1)) / np.std(x_test, axis=1).reshape(-1,1))
            x_test = np.stack([x_test, x_test_p], axis=2)
            j_son_file = open('model_cnn.json','r')  
            loaded_model_j_son = j_son_file.read()
            j_son_file.close()
            loaded_model_cnn = model_from_json(loaded_model_j_son)
            loaded_model_cnn.load_weights("model_cnn.h5")
            loaded_model_cnn.summary()
        if st.button('Predict using cnn'):    
            y_predict = loaded_model_cnn.predict(x_test)[rows:rows1] 
           
            html_temp = """ 
                <div style ="background-color:black;padding:13px"> 
                <h2 style ="color:white;text-align:center;">The predicted value is </h2> 
                </div> 
                """
            st.markdown(html_temp, unsafe_allow_html=True)
            st.write(y_predict)
            if y_predict < 0.7 :
                html_temp = """ 
                <div style ="background-color:pink;padding:10px"> 
                <h3 style ="color:black;text-align:center;">Nope ! </h3> 
                </div> 
                """
            else:
                
                html_temp = """ 
                <div style ="background-color:pink;padding:10px"> 
                <h3 style ="color:black;text-align:center;">Wohoo ! We found an exoplanet for the star !</h3> 
                </div> 
                """
            st.markdown(html_temp, unsafe_allow_html=True)


if page == "About":
    #st.image("exologo.jpeg")
    
    html_temp = """ 
        <div style ="background-color:navy;padding:13px"> 
        <h1 style ="color:white;text-align:center;">About</h1> 
        </div> 
        """
    st.markdown(html_temp, unsafe_allow_html=True)
    main_bg = "about_bg.jpeg"
    main_bg_ext = "jpeg"

    st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
    }}

    </style>
    """,
    unsafe_allow_html=True
    )
    
    #st.markdown("Official documentation of **[Facebook Prophet](https://facebook.github.io/prophet/)**")
    st.markdown("Official documentation of **[Streamlit](https://docs.streamlit.io/en/stable/getting_started.html)**")
    st.write("")
    st.write("Author:")
    st.markdown(""" **[Cicy K Agnes](https://www.linkedin.com/in/cicykagnes/)**""")
    st.markdown(""" **[Akthar Naveed](https://www.linkedin.com/in/akthar-naveed-v-921039201)**""")
    st.markdown("""**[Source code](https://github.com/cicykagnes/exoplanet/blob/main/exo_frnt.py)**""")

    st.write("Created on 09/05/2021")
    st.write("Last updated: **09/05/2021**")