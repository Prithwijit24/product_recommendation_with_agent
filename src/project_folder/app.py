from PIL.Image import Image
import streamlit as st
from PIL import Image
import numpy as np
import subprocess as sb
import os
import cv2
import pycountry

import json
st.markdown("# **Next-Gen Personalized E-commerce Assistant**", width = "stretch")
st.markdown("- This app uses models trained on the publicly available ***UTKFace*** dataset for demonstration purposes only. We :green[do not] store any user :green[images, personal data, or prediction] results. All data is processed temporarily and discarded after use.")
from recommendation_agent import agent_creation_wrapper
from streamlit_product_card import product_card
import json
from ddgs import DDGS
import uuid
st.spinner("loading larger files .... ", show_time = True):
    from main import main
import gdown

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

sb.run(f"mkdir -p data/{st.session_state.session_id}", text = True, shell = True)


if "llm_radio" not in st.session_state:
    st.session_state.llm_radio = None
if "llm_api_key" not in st.session_state:
    st.session_state.llm_api_key = None
if "tavily_api_key" not in st.session_state:
    st.session_state.tavily_api_key = None

with st.sidebar:
    st.session_state.llm_radio = st.radio("Select the domain", ["Openrouter", "Grok"])

    st.session_state.llm_api_key = st.text_input("LLM API KEY", placeholder = "LLM API Key", type = "password")
    if st.session_state.llm_radio == 'Openrouter':
        st.link_button("Generate", url = "https://openrouter.ai/settings/keys")
    else:
         st.link_button("Generate", url = "https://console.groq.com/keys")


    st.session_state.tavily_api_key = st.text_input("TAVILY API KEY", placeholder = "TAVILY API Key", type = "password")
    st.link_button("Generate", "https://app.tavily.com/home") 

st.set_page_config(layout="wide")
if "gender" not in st.session_state:
    st.session_state.gender = None
if "race" not in st.session_state:
    st.session_state.race = None
if "age" not in st.session_state:
    st.session_state.age = 0

if "camera_image" not in st.session_state:
    st.session_state.camera_image = None
if "upload_image" not in st.session_state:
    st.session_state.upload_image = None
if "run_click" not in st.session_state:
    st.session_state.run_click = False
if "reco" not in st.session_state:
    st.session_state.reco = False
if "prediction" not in st.session_state:
    st.session_state.prediction = False
if "imshow" not in st.session_state:
    st.session_state.imshow = None


def face_identification(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    img = image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces):
        x, y, w, h = faces[0]
        new_w = int(w * 1.1)
        new_h = int(h * 1.1)

        new_x = max(0, int(x - (new_w - w) / 2))
        new_y = max(0, int(y - (new_h - h) / 2))
        # Ensure box stays within image bounds
        h_img, w_img, _ = image.shape
        new_x2 = min(w_img, new_x + new_w)
        new_y2 = min(h_img, new_y + new_h)

        # Crop face
        face_crop = image[new_y:new_y2, new_x:new_x2]
    else:
        face_crop = image

    os.makedirs("data", exist_ok=True)
    cv2.imwrite(f"data/{st.session_state.session_id}/photo.jpg", face_crop)

    return face_crop


st.divider()
st.markdown("### :snowflake: Our Models Historical Performance on :orange[Demographic Characteristics] Prediction ")
gender_acc, race_acc, age_acc = st.columns(3)
gender_acc.metric(label = "Gender Prediction Accuracy", value = "97%", border = True)
race_acc.metric(label = "Ethnicity Prediction Accuracy", value = "94%", border = True)
age_acc.metric(label = "Age Prediction Deviation", value = "5.7 years", border = True)

st.divider()
st.markdown("### :snowflake: Now Its Your Turn :relaxed:")
select_media = st.radio("Please select how to upload your photo",options=("upload from device", "open camera"), horizontal=True)


with st.container(border=True, vertical_alignment='center', horizontal_alignment='center'):
    image_col, divider_col, text_col = st.columns([3, 0.01, 4])

    with image_col:

        image_placeholder = st.empty()
        if select_media == "open camera":
            # with st.spinner('Photo is being processed ...'):

            st.session_state.camera_image = image_placeholder.camera_input("Please take your photo")
        else:
            st.session_state.upload_image = image_placeholder.file_uploader("Please upload your photo", label_visibility="visible", )


        if "image_to_use" not in st.session_state:
            st.session_state.image_to_use = None

        if st.session_state.camera_image is not None:
            image = Image.open(st.session_state.camera_image)
            st.session_state.image_to_use = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        elif st.session_state.upload_image is not None:
            image = Image.open(st.session_state.upload_image)
            st.session_state.image_to_use = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

       
        if st.session_state.image_to_use is not None:
            result = face_identification(st.session_state.image_to_use)
            _, im_subcol, _ = st.columns([1,4,1])

            # if result.shape[0] > result.shape[1]:
            new_height = 150
            h, w = result.shape[:2]
            aspect_ratio = w / new_height
            new_width = 150
            resized_img = cv2.resize(result, (new_width, new_height))
            resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
            st.session_state.imshow = resized_img
            image_placeholder.empty()
            im_subcol.image(st.session_state.imshow, caption="***Voila:balloon: !!!  :ok_hand:***")

            docs_placeholder, button_placeholder = image_col.columns([5,1.6])
            docs_placeholder.markdown("Press :red-badge[Run] to see age, race, and gender")
            if button_placeholder.button("Run"):
                st.session_state.run_click = True

    with divider_col:
        st.markdown(
        """
        <div style="
            border-left: 2px solid #ccc;
            height: 300px;
            margin: auto;
        "></div>
        """,
        unsafe_allow_html=True
    )
    
    text_placeholder = st.empty()
    unicode_dict = {
        'gender':{
            'Male':  '♂️', 
            'Female': '♀️'
        },
        'race': ':sparkles:',
        'Male_age': ":boy:" if st.session_state.age <= 30 else ":man:" if st.session_state.age <= 60 else ":older_man:",
        'Female_age': ":girl:" if st.session_state.age <= 30 else ":woman:" if st.session_state.age <= 60 else ":older_woman:"
    }

    if st.session_state.run_click:

        text_col.subheader(":dash: Model's Prediction on Characteristics",divider = "rainbow")
        emoji_placeholder, text_placeholder = text_col.columns([0.6, 10])

        with text_col:
            if not os.path.exists("models"):
                folder_url = "https://drive.google.com/drive/folders/1sYNxiyrP5ExRDepFuGd5Y6RyIsZnC_jm?usp=drive_link"
                with st.spinner("Downloading the models ----", show_time = True):
                    gdown.download_folder(
                        folder_url,
                        output="models",
                        quiet=False,
                        use_cookies=False
                    )

            if st.session_state.run_click and not st.session_state.prediction:
                with st.spinner(text="I am trying to predict the gender ...... ", show_time=True):
                    st.session_state.gender = main(prediction_type = "single", target = "gender", image_path = f"data/{st.session_state.session_id}/photo.jpg")

            emoji_placeholder.markdown(":white_check_mark:")
            text_placeholder.markdown(f"**Predicted Gender (Model Estimate):** **{st.session_state.gender}** {unicode_dict['gender'][st.session_state.gender]}", unsafe_allow_html=True)

            if st.session_state.run_click and not st.session_state.prediction: 
                with st.spinner(text="I am trying to predict the race ...... ", show_time=True):
                    st.session_state.race = main(prediction_type = "single", target = "race", image_path = f"data/{st.session_state.session_id}/photo.jpg")

            emoji_placeholder.markdown(":white_check_mark:")
            text_placeholder.markdown(f"**Predicted Ethnicity (Model Estimate):** **{st.session_state.race}** {unicode_dict['race']}")

            if st.session_state.run_click and not st.session_state.prediction: 
                with st.spinner(text="I am trying to predict the age ...... ", show_time=True):
                    st.session_state.age = int(round(float(main(prediction_type = "single", target = "age", image_path = f"data/{st.session_state.session_id}/photo.jpg"))))

            emoji_placeholder.markdown(":white_check_mark:")
            text_placeholder.markdown(f"**Predicted Age (Model Estimate):** **{st.session_state.age}** {unicode_dict[st.session_state.gender+'_age']}")
            
            st.session_state.prediction = True
            st.space("medium")
            
            docs_placeholder_1, button_placeholder_1 = text_col.columns([4,2])
            docs_placeholder_1.markdown("Press :red-badge[Recommend] button for your reco")
            if button_placeholder_1.button("Recommend"):
                st.session_state.reco = True
                docs_placeholder_1.snow()
    
st.divider()


if st.session_state.reco:
   
    st.markdown("### :snowflake: Recommendation Section")
    if 'pc' not in st.session_state:
        st.session_state.pc = None
    if 'vibe' not in st.session_state:
        st.session_state.vibe = None
    if 'budget' not in st.session_state:
        st.session_state.budget = None
    if 'location' not in st.session_state:
        st.session_state.location = None
    if 'text' not in st.session_state:
        st.session_state.text = None

    product_placeholder, budget_placeholder, location_placeholder, text_placeholder, reco_button_placeholder = st.columns([1, 1, 1, 4, 0.7])
    st.session_state.pc = [product_placeholder.selectbox("Product_Category", ['Clothing', 'Technology', 'Home Decor', 'Books', 'Grocery', 'other'])]
    if 'other' in st.session_state.pc:
        st.session_state.pc.remove('other')
        pc_str = product_placeholder.text_input(label = "mention the catgory", placeholder = 'e.g. Games, Shoes. If you want to mention multiple categories put a comma in between')
        if ',' not in pc_str:
            st.session_state.pc.append(pc_str)
        else:
            pc_list = [i.strip() for i in pc_str.split(',')]

    product_placeholder.empty()
    countries = sorted([country.name for country in pycountry.countries])
    # st.session_state.vibe = vibe_placeholder.selectbox(label = "Select Vibe", [''])
    st.session_state.budget = budget_placeholder.selectbox(label = "Budget", options=['High', 'Medium', 'Low'])
    st.session_state.location = location_placeholder.selectbox(label = "Location", options=countries, index = None, placeholder="India")
    st.session_state.text = text_placeholder.text_input("If you want to add something", placeholder="add something extra")


    if "reco_button" not in st.session_state:
        st.session_state.reco_button = False

    if st.session_state.pc and st.session_state.budget and st.session_state.location:
        st.session_state.reco_button = False
        reco_button_placeholder.text("")
        reco_button_placeholder.text("")
        
        if st.session_state.llm_api_key and st.session_state.tavily_api_key:
            st.session_state.reco_button = reco_button_placeholder.button("Go", disabled = False)
        else:
            st.session_state.reco_button = reco_button_placeholder.button("Go", disabled = True)


  
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if "response" not in st.session_state:
        st.session_state.response = None
    
    st.session_state.agent = agent_creation_wrapper(st.session_state.age, st.session_state.race, st.session_state.gender, st.session_state.location, st.session_state.llm_radio, st.session_state.llm_api_key, st.session_state.tavily_api_key)

    if st.session_state.reco_button:
        with st.spinner("Thinking", show_time = True):
            st.session_state.response = st.session_state.agent.invoke({'messages': 
                                              [{
                                                  'role': 'user',
                                                  'content': f"Recommend me 3 most relevant products from {st.session_state.pc} category. budget is {st.session_state.budget}. {st.session_state.text}. If category is Book, show autors' name too"
                                                  }]})['messages'][-1].content
            st.session_state.response = json.loads(st.session_state.response.strip().replace("```json", "").replace("```", ""))
    

    if st.session_state.response: 
        with st.container(border = True):
            for i in range(len(st.session_state.response)):
                product_card(
                    product_name = st.session_state.response[i]['product_name'],
                    description = st.session_state.response[i]['product_description'],
                    price = st.session_state.response[i]['product_price'],
                    product_image =  DDGS().images(st.session_state.response[i]['product_name'], max_results = 5)[0]['image'],
                    key=f"card_{i}",
                    picture_position="left",
                    image_aspect_ratio="8/8",
                    image_object_fit="content"
                    )

with st.container():
    cols = st.columns([10, 2])
    cols[0].markdown("Want to Clear the page --- Want to delete the data")
    with cols[1]:
        if st.button("Clean"):
            sb.run(f"rm -rf data/{st.session_state.session_id}", text = True, shell = True)
            st.session_state.clear()


# st.session_state.clear()
# st.balloons()






