"""Next-Gen Personalized E-commerce: face → demographics → agentic skincare routine."""

import logging
import os
import subprocess as sb
import sys
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

import cv2
import gdown
import numpy as np
import streamlit as st
from main import main
from PIL import Image

from project_folder.agentic import get_questionnaire, orchestrate

st.set_page_config(layout="wide")

st.markdown("# **Next-Gen Personalized E-commerce Application**", width="stretch")
st.markdown(
    "- This app uses models trained on the publicly available ***UTKFace*** dataset for "
    "demonstration purposes only. We :green[do not] store any user :green[images, personal data, "
    "or prediction] results. All data is processed temporarily and discarded after use."
)

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

sb.run(f"mkdir -p data/{st.session_state.session_id}", text=True, shell=True)

for state_key in ("gender", "race"):
    if state_key not in st.session_state:
        st.session_state[state_key] = None
if "age" not in st.session_state:
    st.session_state.age = 0
if "camera_image" not in st.session_state:
    st.session_state.camera_image = None
if "upload_image" not in st.session_state:
    st.session_state.upload_image = None
if "run_click" not in st.session_state:
    st.session_state.run_click = False
if "run_skincare" not in st.session_state:
    st.session_state.run_skincare = False
if "prediction" not in st.session_state:
    st.session_state.prediction = False
if "imshow" not in st.session_state:
    st.session_state.imshow = None


def face_identification(image):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces):
        x, y, w, h = faces[0]
        new_w = int(w * 1.1)
        new_h = int(h * 1.1)
        new_x = max(0, int(x - (new_w - w) / 2))
        new_y = max(0, int(y - (new_h - h) / 2))
        h_img, w_img, _ = image.shape
        new_x2 = min(w_img, new_x + new_w)
        new_y2 = min(h_img, new_y + new_h)
        face_crop = image[new_y:new_y2, new_x:new_x2]
    else:
        face_crop = image

    os.makedirs("data", exist_ok=True)
    cv2.imwrite(f"data/{st.session_state.session_id}/photo.jpg", face_crop)
    return face_crop


def age_range(age: int) -> str:
    if age < 18:
        return "0-17"
    if age < 25:
        return "18-24"
    if age < 35:
        return "25-34"
    if age < 45:
        return "35-44"
    if age < 55:
        return "45-54"
    if age < 65:
        return "55-64"
    return "65+"


st.divider()
st.markdown(
    "### :snowflake: Model Historical Performance on :orange[Demographic Prediction] "
)
gender_acc, race_acc, age_acc = st.columns(3)
gender_acc.metric(label="Gender Prediction Accuracy", value="97%", border=True)
race_acc.metric(label="Ethnicity Prediction Accuracy", value="94%", border=True)
age_acc.metric(label="Age Prediction Deviation", value="5.7 years", border=True)

st.divider()
st.markdown("### :snowflake: Now It's Your Turn :relaxed:")
select_media = st.radio(
    "Please select how to upload your photo",
    options=("upload from device", "open camera"),
    horizontal=True,
)

with st.container(border=True, vertical_alignment="center", horizontal_alignment="center"):
    image_col, divider_col, text_col = st.columns([3, 0.01, 4])

    with image_col:
        image_placeholder = st.empty()
        if select_media == "open camera":
            st.session_state.camera_image = image_placeholder.camera_input(
                "Please take your photo"
            )
        else:
            st.session_state.upload_image = image_placeholder.file_uploader(
                "Please upload your photo",
                label_visibility="visible",
            )

        if "image_to_use" not in st.session_state:
            st.session_state.image_to_use = None

        if st.session_state.camera_image is not None:
            image = Image.open(st.session_state.camera_image)
            st.session_state.image_to_use = cv2.cvtColor(
                np.array(image), cv2.COLOR_RGB2BGR
            )
        elif st.session_state.upload_image is not None:
            image = Image.open(st.session_state.upload_image)
            st.session_state.image_to_use = cv2.cvtColor(
                np.array(image), cv2.COLOR_RGB2BGR
            )

        if st.session_state.image_to_use is not None:
            result = face_identification(st.session_state.image_to_use)
            _, im_subcol, _ = st.columns([1, 4, 1])
            resized_img = cv2.resize(result, (150, 150))
            resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
            st.session_state.imshow = resized_img
            image_placeholder.empty()
            im_subcol.image(
                st.session_state.imshow, caption="***Voila:balloon: !!!  :ok_hand:***"
            )

            pred_placeholder, button_placeholder = image_col.columns([5, 1.6])
            pred_placeholder.markdown(
                "Press :red-badge[Run] to see age, race, and gender"
            )
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
            unsafe_allow_html=True,
        )

    if st.session_state.run_click:
        with text_col:
            text_col.subheader(
                ":dash: Model's Prediction on Characteristics", divider="rainbow"
            )

            if not os.path.exists("models"):
                text_col.caption("Downloading models from Drive once (first run only) ...")
                with st.spinner("Downloading the models ----", show_time=True):
                    gdown.download_folder(
                        "https://drive.google.com/drive/folders/1sYNxiyrP5ExRDepFuGd5Y6RyIsZnC_jm?usp=drive_link",
                        output="models",
                        quiet=False,
                        use_cookies=False,
                    )

            photopath = f"data/{st.session_state.session_id}/photo.jpg"

            # Make predictions if not already done
            if not st.session_state.prediction:
                with st.spinner("I am trying to predict the gender ...... ", show_time=True):
                    st.session_state.gender = main(
                        prediction_type="single", target="gender", image_path=photopath
                    )

                with st.spinner("I am trying to predict the race ...... ", show_time=True):
                    st.session_state.race = main(
                        prediction_type="single", target="race", image_path=photopath
                    )

                with st.spinner("I am trying to predict the age .......", show_time=True):
                    st.session_state.age = round(
                            float(
                                main(
                                    prediction_type="single",
                                    target="age",
                                    image_path=photopath,
                                )
                            )
                        )
                st.session_state.prediction = True

            # Always display predictions (persists across rerenders)
            if st.session_state.prediction:
                text_col.markdown(
                    f"**Predicted Gender (Model Estimate):** {st.session_state.gender}"
                )
                text_col.markdown(
                    f"**Predicted Ethnicity (Model Estimate):** {st.session_state.race}"
                )
                text_col.markdown(
                    f"**Predicted Age (Model Estimate):** {st.session_state.age} years"
                )


st.divider()

if st.session_state.prediction and text_col.button("Get My Skincare Routine"):
                    st.session_state.run_skincare = True

if st.session_state.run_skincare:
    st.markdown("### :snowflake: Skincare Recommendation Section")
    st.markdown(
        "- Based on the predicted demographics we ask a few questions, then the "
        "**agentic orchestrator** researches likely skin considerations, searches for real "
        "products, and enforces a **deterministic safety gate** before returning a routine "
        "(with disclaimer)."
    )

    questionnaire = get_questionnaire()
    answers: dict = {}

    q_cols = st.columns(3)
    for i, q in enumerate(questionnaire):
        with q_cols[i % 3]:
            if q["id"] == "sensitivities":
                answers[q["id"]] = st.multiselect(q["label"], q["options"], default=["None"])
            else:
                answers[q["id"]] = st.selectbox(q["label"], q["options"])

    answers["pregnant"] = st.radio(
        "Are you currently pregnant?", ["no", "yes"], horizontal=True
    )

    if st.button("Run Skincare Orchestrator", type="primary"):
        demographics = {
            "age_range": age_range(st.session_state.age),
            "sex": {"value": "M" if st.session_state.gender == "Male" else "F"},
            "race": {"value": st.session_state.race},
        }
        with st.spinner(
            "Researching, searching products, checking safety ......", show_time=True
        ):
            st.session_state.skincare_result = orchestrate(
                demographics, answers, session_id=f"ui-{st.session_state.session_id}"
            )

    if "skincare_result" in st.session_state:
        st.divider()
        st.markdown("#### :sparkles: Your Personalized Routine")
        result = st.session_state.skincare_result

        if not result.get("routine"):
            st.info("No products survived the safety gate for this profile.")
        else:
            with st.container(border=True):
                concerns = result.get('concerns_addressed', [])
                if concerns:
                    st.markdown("**🎯 :blue[Concerns Addressed:]** " + ", ".join(f"*{c}*" for c in concerns))

                for i, item in enumerate(result.get("routine", [])[:3]):  # Only show 3 products
                    with st.container(border=True):
                        # Two-column layout: text on left, image on right
                        text_col, img_col = st.columns([3, 1])

                        with text_col:
                            # Product name - smaller and bold
                            st.markdown(f"**:package: {item.get('product_name', 'Unknown Product')}**")
                            # Price directly under name
                            st.markdown(f"**:green[Price: {item.get('price', 'Price unavailable')}]**")

                            # Ingredient with color
                            st.markdown(f"**🔬 Key Ingredient:** :orange[{item.get('ingredient', 'N/A')}]")

                            # Reasoning with formatting
                            reasoning = item.get('reasoning', '')
                            if reasoning:
                                st.markdown("**💡 Why this product:**")
                                # Clean up the reasoning text - remove wrapping asterisks
                                cleaned_reasoning = reasoning.strip()
                                while cleaned_reasoning.startswith('*') and cleaned_reasoning.endswith('*'):
                                    cleaned_reasoning = cleaned_reasoning[1:-1].strip()
                                # Render with italic styling using HTML
                                st.markdown(
                                    f"<div style='font-style: italic; color: #666666; "
                                    f"background-color: #f8f9fa; padding: 10px; "
                                    f"border-radius: 5px; border-left: 3px solid #4a90d9;'>"
                                    f"{cleaned_reasoning}</div>",
                                    unsafe_allow_html=True,
                                )

                            # Product URL - aligned right with high contrast
                            url = str(item.get("url") or "")
                            if url.startswith("http"):
                                st.markdown("<br>", unsafe_allow_html=True)  # Breathing space
                                btn_col1, btn_col2, btn_col3 = st.columns([2, 1, 1])
                                with btn_col3:
                                    st.link_button("🛒 Open Product", url, type="primary", help=f"View {item.get('product_name', 'product')} online")

                        with img_col:
                            # Product image on the right - height matches content
                            image_url = item.get("image_url", "")
                            if image_url and isinstance(image_url, str) and image_url.startswith("http"):
                                # Use CSS to make image height match content
                                st.markdown(
                                    f'<div style="display: flex; justify-content: center; align-items: center; height: 100%; min-height: 250px; background-color: #f8f9fa; border-radius: 8px; padding: 10px;">'
                                    f'<img src="{image_url}" style="max-width: 100%; max-height: 300px; object-fit: contain; border-radius: 8px;">'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.markdown(
                                    '<div style="display: flex; justify-content: center; align-items: center; height: 100%; min-height: 250px; background-color: #f8f9fa; border-radius: 8px; padding: 10px;">'
                                    '<span style="color: #999;">🖼️ Image not available</span>'
                                    '</div>',
                                    unsafe_allow_html=True,
                                )

                        st.markdown("---")

        st.caption(str(result.get("disclaimer", "")))
        if result.get("error"):
            st.warning(f"⚠️ Partial run — {result['error']}")



with st.container():
    cols = st.columns([10, 2])
    cols[0].markdown("Want to Clear the page --- Want to delete the data")
    with cols[1]:
        if st.button("Clean"):
            sb.run(f"rm -rf data/{st.session_state.session_id}", text=True, shell=True)
            st.session_state.clear()