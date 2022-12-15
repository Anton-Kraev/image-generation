import io
import streamlit as st
from PIL import Image


def image_uploader(label='', tooltip='', resize=True):
    uploaded_file = st.file_uploader(label=label, help=tooltip, type=['png', 'jpg', 'jpeg'])
    if resize and uploaded_file:
        return Image.open(uploaded_file).resize((512, 512))
    return uploaded_file


with st.sidebar:
    option = st.selectbox('Select mode of app', ('Text-to-image', 'Image-to-image', 'Inpainting'))

    settings, advanced_settings = st.tabs(['Settings', 'Advanced settings'])
    with settings:
        prompt = st.text_area('Prompt')
        if option == 'Image-to-image':
            init_img = image_uploader(label='Upload initial image', resize=False)
        elif option == 'Inpainting':
            tooltip = 'In order not to lose quality, it is advisable to upload square images, ideally 512x512'
            init_img = image_uploader(label='Upload initial image', tooltip=tooltip)
            mask_img = image_uploader(label='Upload mask image', tooltip=tooltip)
    with advanced_settings:
        negative_prompt = st.text_area('Negative prompt')
        guidance_scale = st.slider('Guidance scale', 0., 15., 7.5, step=0.5)
        num_inference_steps = st.slider('Number of inference steps', 1, 100, 50)
        num_images_per_prompt = st.slider('Number of result images', 1, 8, 1)

'You selected: ', option
