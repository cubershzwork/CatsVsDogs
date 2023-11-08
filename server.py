import streamlit as st
from fastai.vision.all import *
import urllib.request

st.title("Cat verses Dog Classifier")
st.text("Built by Shawn Zhuang")

def label_func(f): return f[0].isupper()
# Load our pre-trained model
model = load_learner('my_model.pkl')

# Define a function to make predictions
def predict(image):
    img = PILImage.create(image)
    pred_class, pred_idx, outputs = model.predict(img)
    likelihood_is_cat = outputs[1].item()
    print(likelihood_is_cat)
    if likelihood_is_cat == 1:
        return "cat"
    elif likelihood_is_cat < 0.01:
        return "dog"
    else:
        return "Not sure... try another picture"

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None: # If we upload a file
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"): # If button is pressed
        prediction = predict(uploaded_file)
        st.write(prediction)


# Open terminal on the bottom
# Type streamlit run server.py