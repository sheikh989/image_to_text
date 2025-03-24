import streamlit as st
import requests
from huggingface_hub import InferenceClient
from PIL import Image


# Initialize Hugging Face inference client
client = InferenceClient(
    provider="hf-inference",
    api_key="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",
)

# Streamlit UI
st.markdown("## Text Generator from Image")
img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if img is not None:
    # Display uploaded image
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Convert image to bytes for API
    image_bytes = img.read()

    if st.button("Generate Text"):
        # Send image bytes to API
        response = client.image_to_text(image_bytes, model="Salesforce/blip-image-captioning-base")

        # Extract only the text response
        if isinstance(response, dict) and "generated_text" in response:
            caption = response["generated_text"]
        else:
            caption = "No caption generated."

        # Display response text only
        st.write("### Generated Caption:")
        st.write(caption)
