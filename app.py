import io
import streamlit as st
from PIL import Image
from model import Model


def image_uploader(label, option, tooltip='', resize=True):
    uploaded_file = st.file_uploader(label=label, help=tooltip, type=['png', 'jpg', 'jpeg'], key=option + label)
    if resize and uploaded_file:
        return Image.open(uploaded_file).resize((512, 512))
    return uploaded_file


with st.sidebar:
    task = st.selectbox('Select task', ('text2img', 'img2img', 'inpaint'))
    if task not in st.session_state.keys():
        st.session_state[task] = {}
    state = st.session_state[task]

    settings, advanced_settings = st.tabs(['Settings', 'Advanced settings'])

    with settings:
        state['prompt'] = st.text_area(label='Prompt', value=state.get('prompt') or '', key=task + 'Prompt')
        if task == 'img2img':
            state['init_img'] = image_uploader('Upload initial image', task, resize=False)
        elif task == 'inpaint':
            tooltip = 'In order not to lose quality, it is advisable to upload square images, ideally 512x512'
            state['init_img'] = image_uploader('Upload initial image', task, tooltip=tooltip)
            state['mask_img'] = image_uploader('Upload mask image', task, tooltip=tooltip)

    with advanced_settings:
        state['negative_prompt'] = st.text_area(label='Negative prompt', value=state.get('negative_prompt') or '',
                                                key=task + 'Negative prompt')
        state['guidance_scale'] = st.slider(label='Guidance scale', min_value=0., max_value=15.,
                                            value=state.get('guidance_scale') or 7.5, step=0.5,
                                            key=task + 'Guidance scale')
        state['num_inference_steps'] = st.slider(label='Number of inference steps', min_value=1, max_value=100,
                                                 value=state.get('num_inference_steps') or 20,
                                                 key=task + 'Number of inference steps')
        state['num_images_per_prompt'] = st.slider(label='Number of result images', min_value=1, max_value=8,
                                                   value=state.get('num_images_per_prompt') or 1,
                                                   key=task + 'Number of result images')
    st.button(label='Generate image(s)', on_click=None, disabled=False)

st.caption('In all samples used *Guidance scale = 7.5* and *Number of inference steps = 50*')
st.title(f'Sample of *{task}* usage')
if task == 'text2img':
    sample_prompt = 'An astronaut riding a horse'
elif task == 'img2img':
    sample_prompt = 'A fantasy landscape, trending on artstation'
else:
    sample_prompt = 'Face of a yellow cat, high resolution, sitting on a park bench'
st.subheader(f'Prompt = {sample_prompt}')
if task == 'img2img':
    st.image('./samples/img2img/init_image.jpg', 'init_image')
elif task == 'inpaint':
    st.image('./samples/inpaint/init_image.png', 'init_image')
    st.image('./samples/inpaint/mask_image.png', 'mask_image')
st.subheader('Result')
st.image(f'./samples/{task}/result.png')

