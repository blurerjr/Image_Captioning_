import streamlit as st
import torch
from PIL import Image
import io
from transformers import AutoProcessor, BlipForConditionalGeneration

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="BLIP Image Captioning App", page_icon="📝", layout="centered")
st.title("📝 Image Captioning with BLIP")
st.markdown("---") # Visual separator

# Inject custom CSS for more fine-grained control
st.markdown(
    """
    <style>
    /* Main App Background (if config.toml isn't enough, this is more forceful) */
    .stApp {
        background-color: #1E1E1E; /* A very dark grey */
        color: #FAFAFA; /* Light text color for the main app */
    }

    /* Adjusting header/title color if needed */
    h1, h2, h3, h4, h5, h6 {
        color: #E0E0E0; /* Slightly lighter grey for headers */
    }

    /* Style for the "Generated Captions" subheader */
    .caption-subheader {
        color: #90EE90; /* Light green, matching primaryColor */
        font-size: 1.8em; /* Slightly larger */
        font-weight: bold;
        margin-top: 1em;
        margin-bottom: 0.5em;
    }
    
/* Style for the labels of the text areas (Caption 1, Caption 2, etc.) */
    /* This targets the internal label of st.text_area */
    .stTextArea label {
        color: #FFFFFF !important; /* Pure white for labels */
        font-size: 1.1em;
        font-weight: bold;
    }
    /* Style for the text area itself */
    .stTextArea textarea {
        background-color: #2D2D2D !important; /* Keep background dark grey for contrast */
        color: #FFFFFF !important; /* Pure white for text inside the text area */
        border: 1px solid #666666; /* Slightly lighter border for visibility */
        border-radius: 5px;
        padding: 10px;
    }


    /* Adjust sidebar background to match if needed, though config.toml should handle base */
    [data-testid="stSidebar"] {
        background-color: #2D2D2D; /* Match secondaryBackgroundColor */
        color: #FAFAFA;
    }

    /* Style for the 'Running on: **DEVICE**' info in sidebar */
    [data-testid="stSidebar"] .st-ea { /* Targets the specific element Streamlit uses for st.info */
        color: #FAFAFA;
        background-color: #3A3A3A; /* Slightly different shade for info box */
        border-radius: 5px;
        padding: 5px;
        margin-top: 10px;
        margin-bottom: 10px;
    }

    /* Buttons style for better dark theme look */
    .stButton>button {
        background-color: #90EE90; /* Light green background */
        color: #1E1E1E; /* Dark text on button */
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #76C776; /* Slightly darker green on hover */
        color: #1E1E1E;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# --- Model Loading (Cached for Performance) ---
@st.cache_resource
def load_blip_model():
    """
    Loads the BLIP model and processor, caching them for efficient Streamlit reruns.
    """
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_blip_model()

# --- Device Configuration ---
# Check if CUDA is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
st.sidebar.info(f"Running on: **{device.type.upper()}**")

# --- Caption Generation Function ---
def generate_blip_captions(image_pil, max_length=50, num_beams=4, num_return_sequences=3):
    """
    Generates captions for a given PIL Image using the BLIP model.
    """
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
    # Display the uploaded image
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
        # Use HTML for a custom-styled subheader
        st.markdown('<p class="caption-subheader">Here are the generated captions:</p>', unsafe_allow_html=True)
        
        # --- NEW CODE FOR CAPTION DISPLAY ---
        # Create columns to arrange the text areas
        # Adjust column widths based on how many captions you typically generate
        # For 3 captions, [1, 1, 1] would make them equal width across.
        # For better vertical stacking, no columns are strictly needed, but it helps align.
        
        # If you want them to stack vertically, simply loop without columns:
        for i, caption in enumerate(captions):
            st.text_area(f"Caption {i+1}", value=caption, height=70, key=f"caption_output_{i}", disabled=True)
        # --- END NEW CODE ---

else:
    st.info("Upload an image to begin captioning!")

st.markdown("---")
st.markdown("""
This application utilizes the **Salesforce BLIP (Bootstrapping Language-Image Pre-training)** model from Hugging Face Transformers.
BLIP is a powerful model designed for unified vision-language understanding and generation, providing robust image captioning capabilities.
""")
