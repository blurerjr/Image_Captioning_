import streamlit as st
import torch
from PIL import Image
import io
from transformers import AutoProcessor, BlipForConditionalGeneration

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="BLIP Image Captioning App", page_icon="📝", layout="centered")
st.title("📝 Image Captioning with BLIP")
st.markdown("---") # Visual separator

# --- Theme Management ---
# Initialize theme in session state if not already set
if "theme" not in st.session_state:
    st.session_state.theme = "dark" # Default to dark theme

# Define CSS for dark and light themes
DARK_THEME_CSS = """
    <style>
    .stApp {
        background-color: #1E1E1E; /* Very dark grey */
        color: #FAFAFA; /* Light text for main app */
    }
    h1, h2, h3, h4, h5, h6 {
        color: #E0E0E0; /* Slightly lighter grey for headers */
    }
    .caption-subheader {
        color: #90EE90; /* Light green, accent color */
        font-size: 1.8em;
        font-weight: bold;
        margin-top: 1em;
        margin-bottom: 0.5em;
    }
    .stTextArea label {
        color: #FFFFFF !important; /* Pure white for labels */
        font-size: 1.1em;
        font-weight: bold;
    }
    .stTextArea div[data-baseweb="textarea"] textarea {
        background-color: #3A3A3A !important; /* Darker grey for text area background */
        color: #FFFFFF !important; /* Pure white for text inside */
        border: 1px solid #555555;
        border-radius: 5px;
        padding: 10px;
    }
    .stTextArea div[data-baseweb="textarea"] textarea:disabled {
        opacity: 0.9;
    }
    [data-testid="stSidebar"] {
        background-color: #2D2D2D; /* Dark sidebar */
        color: #FAFAFA; /* Sidebar text color */
    }
    [data-testid="stSidebar"] .st-ea {
        color: #FAFAFA; /* st.info text color */
        background-color: #3A3A3A;
        border-radius: 5px;
        padding: 5px;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .stButton>button {
        background-color: #90EE90; /* Light green button */
        color: #1E1E1E;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #76C776;
        color: #1E1E1E;
    }
    </style>
"""

LIGHT_THEME_CSS = """
    <style>
    .stApp {
        background-color: #FFFFFF; /* Pure white */
        color: #333333; /* Dark text for main app */
    }
    h1, h2, h3, h4, h5, h6 {
        color: #222222; /* Darker grey for headers */
    }
    .caption-subheader {
        color: #008000; /* Darker green, accent color */
        font-size: 1.8em;
        font-weight: bold;
        margin-top: 1em;
        margin-bottom: 0.5em;
    }
    .stTextArea label {
        color: #333333 !important; /* Force dark text for labels in light mode */
        font-size: 1.1em;
        font-weight: bold;
    }
    .stTextArea div[data-baseweb="textarea"] textarea {
        background-color: #F0F0F0 !important; /* Light grey for text area background */
        color: #333333 !important; /* Force dark text inside in light mode */
        border: 1px solid #BBBBBB;
        border-radius: 5px;
        padding: 10px;
    }
    .stTextArea div[data-baseweb="textarea"] textarea:disabled {
        opacity: 0.9; /* Maintain good visibility */
    }
    [data-testid="stSidebar"] {
        background-color: #F5F5F5; /* Light sidebar */
        color: #333333; /* Sidebar text color in light mode */
    }
    [data-testid="stSidebar"] .st-ea {
        color: #333333; /* st.info text color in light mode */
        background-color: #E0E0E0;
        border-radius: 5px;
        padding: 5px;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .stButton>button {
        background-color: #008000; /* Green button */
        color: #FFFFFF; /* White text on button */
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #006400; /* Darker green on hover */
        color: #FFFFFF;
    }
    </style>
"""

# Inject the appropriate CSS based on the current theme
if st.session_state.theme == "dark":
    st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)
else:
    st.markdown(LIGHT_THEME_CSS, unsafe_allow_html=True)

# --- Theme Toggle Button ---
def toggle_theme():
    if st.session_state.theme == "dark":
        st.session_state.theme = "light"
    else:
        st.session_state.theme = "dark"

# Place the toggle button in the sidebar for easy access
st.sidebar.button(
    f"Switch to {'Light' if st.session_state.theme == 'dark' else 'Dark'} Mode",
    on_click=toggle_theme
)

# --- Model Loading (Cached for Performance) ---
@st.cache_resource
def load_blip_model():
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_blip_model()

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
st.sidebar.info(f"Running on: **{device.type.upper()}**")

# --- Caption Generation Function ---
def generate_blip_captions(image_pil, max_length=50, num_beams=4, num_return_sequences=3):
    if image_pil.mode != "RGB":
        image_pil = image_pil.convert("RGB")

    inputs = processor(images=image_pil, return_tensors="pt").to(device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_length,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        early_stopping=True
    )

    preds = processor.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]

    return tuple(preds)

# --- Streamlit UI ---
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

# Generation parameters in sidebar for user control
st.sidebar.header("Generation Settings")
selected_max_length = st.sidebar.slider("Max Caption Length", min_value=10, max_value=100, value=50, step=5)
selected_num_beams = st.sidebar.slider("Number of Beams", min_value=1, max_value=10, value=4, step=1)
selected_num_return_sequences = st.sidebar.slider("Number of Captions to Generate", min_value=1, max_value=5, value=3, step=1)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    if st.button("Generate Captions"):
        st.write("Generating captions...")
        with st.spinner('Thinking... This might take a moment depending on the image and settings.'):
            captions = generate_blip_captions(
                image,
                max_length=selected_max_length,
                num_beams=selected_num_beams,
                num_return_sequences=selected_num_return_sequences
            )
        
        st.success("Captions Generated!")
        st.markdown('<p class="caption-subheader">Here are the generated captions:</p>', unsafe_allow_html=True)
        
        for i, caption in enumerate(captions):
            st.text_area(f"Caption {i+1}", value=caption, height=70, key=f"caption_output_{i}", disabled=True)

else:
    st.info("Upload an image to begin captioning!")

st.markdown("---")
st.markdown("""
This application utilizes the **Salesforce BLIP (Bootstrapping Language-Image Pre-training)** model from Hugging Face Transformers.
BLIP is a powerful model designed for unified vision-language understanding and generation, providing robust image captioning capabilities.
""")
