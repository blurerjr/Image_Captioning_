import streamlit as st
import gdown
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.layers import Layer, Dense, concatenate  # for BahdanauAttention
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import json
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# --- 1. Define Custom Layers (REQUIRED for model loading) ---
class BahdanauAttention(Layer):
    def __init__(self, units, **kwargs):
        super(BahdanauAttention, self).__init__(**kwargs)
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def build(self, input_shape):
        if not (isinstance(input_shape, (list, tuple)) and len(input_shape) == 2):
            raise ValueError(
                f"BahdanauAttention layer expects 2 input shapes (list/tuple), "
                f"but received: {input_shape}."
            )
        features_shape, hidden_shape = input_shape
        self.W1.build((None, features_shape[-1]))
        self.W2.build((None, hidden_shape[-1]))
        self.V.build((None, self.W1.units))
        super(BahdanauAttention, self).build(input_shape)

    def call(self, inputs):
        features, hidden = inputs
        hidden_with_time_axis = K.expand_dims(hidden, 1)
        score = self.V(K.tanh(self.W1(features) + self.W2(hidden_with_time_axis)))
        attention_weights = K.softmax(score, axis=1)
        context_vector = attention_weights * features
        context_vector = K.sum(context_vector, axis=1)
        return context_vector, attention_weights

    def get_config(self):
        config = super(BahdanauAttention, self).get_config()
        config.update({'units': self.W1.units})
        return config

# --- 2. Load Saved Components (at the beginning of your Streamlit app) ---
@st.cache_resource # Cache the model and tokenizer loading for performance
def load_assets():
    # Google Drive URL for the Keras model - USING DIRECT DOWNLOAD LINK FORMAT
    # Extracted FILE_ID from your provided link: 1JlXXYzEHIhZd5V9ElF77KnoZyRYRs-ig
    drive_url = "https://drive.google.com/uc?export=download&id=1JlXXYzEHIhZd5V9ElF77KnoZyRYRs-ig"
    local_model_path = "temp_model.keras" # Temporary path on the Streamlit server

    # GitHub base URL for other resources (this part is correct)
    github_base_url = "https://raw.githubusercontent.com/blurerjr/Image_Captioning_/refs/heads/master/"

    # --- Download the model file from Google Drive ---
    st.info(f"Downloading model from Google Drive (using direct link format)...")
    try:
        gdown.download(drive_url, local_model_path, quiet=False) # quiet=False shows download progress
        
        # --- IMPORTANT DEBUGGING STEP ---
        # Verify if the file was actually downloaded
        if not os.path.exists(local_model_path):
            st.error(f"Error: Model file '{local_model_path}' was not found after download attempt!")
            st.warning("This likely means the Google Drive download failed. Check permissions or the link validity.")
            raise FileNotFoundError(f"Model file not created at {local_model_path}")
        # Optionally print file size
        # st.info(f"Downloaded model size: {os.path.getsize(local_model_path) / (1024*1024):.2f} MB")
        st.success("Model downloaded from Google Drive!")
    except Exception as e:
        st.error(f"Failed to download model from Google Drive. Error: {e}")
        st.warning("Please ensure the Google Drive file is shared as 'Anyone with the link' and the ID is correct. "
                   "Sometimes, Google Drive blocks large file downloads from remote servers.")
        raise Exception("Model download failed")

    # Load model
    caption_model = load_model(
        local_model_path,
        custom_objects={'BahdanauAttention': BahdanauAttention}
    )

    # --- Download and load tokenizer from GitHub ---
    tokenizer_github_url = os.path.join(github_base_url, "tokenizer.pkl")
    local_tokenizer_path = "tokenizer.pkl"
    st.info(f"Downloading tokenizer from GitHub...")
    try:
        gdown.download(tokenizer_github_url, local_tokenizer_path, quiet=True)
        with open(local_tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
        st.success("Tokenizer downloaded from GitHub!")
    except Exception as e:
        st.error(f"Failed to download tokenizer from GitHub. Error: {e}")
        st.warning(f"Please ensure tokenizer.pkl exists directly in your GitHub repo root: {tokenizer_github_url}")
        raise Exception("Tokenizer download failed")
    os.remove(local_tokenizer_path) # Clean up the downloaded file

    # --- Download and load config from GitHub ---
    config_github_url = os.path.join(github_base_url, "model_config.json")
    local_config_path = "model_config.json"
    st.info(f"Downloading model config from GitHub...")
    try:
        gdown.download(config_github_url, local_config_path, quiet=True)
        with open(local_config_path, 'r') as f:
            model_config = json.load(f)
        st.success("Model config downloaded from GitHub!")
    except Exception as e:
        st.error(f"Failed to download model config from GitHub. Error: {e}")
        st.warning(f"Please ensure model_config.json exists directly in your GitHub repo root: {config_github_url}")
        raise Exception("Model config download failed")
    os.remove(local_config_path) # Clean up the downloaded file

    # Extract parameters from config
    max_length = model_config['max_length']
    vocab_size = model_config['vocab_size']
    D_CNN_features = model_config['D_CNN_features']
    num_image_regions = model_config['num_image_regions']

    # Initialize the CNN feature extractor (DenseNet201)
    base_cnn = DenseNet201(include_top=False, weights='imagenet')
    feature_extractor_model = Model(inputs=base_cnn.input, outputs=base_cnn.output)

    return caption_model, tokenizer, max_length, D_CNN_features, num_image_regions, feature_extractor_model, vocab_size

# Load all assets once when the app starts
caption_model, tokenizer, max_length, D_CNN_features, num_image_regions, feature_extractor_model, vocab_size = load_assets()

# --- 3. Helper Functions (No changes needed here from previous full code) ---

def preprocess_image_for_attention(image_raw, feature_extractor_model_loaded, img_size=224):
    """
    Loads, preprocesses, and extracts both flat and spatial features for a single image.
    Takes raw image data (bytes or numpy array) as input.
    """
    if isinstance(image_raw, bytes):
        img_pil = tf.keras.preprocessing.image.load_img(tf.io.BytesIO(image_raw))
    else:
        img_pil = tf.keras.preprocessing.image.array_to_img(image_raw, scale=False)

    img_pil = img_pil.resize((img_size, img_size))
    img = img_to_array(img_pil)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    spatial_feature = feature_extractor_model_loaded.predict(img, verbose=0)
    spatial_feature_map = spatial_feature.reshape(spatial_feature.shape[0], -1, spatial_feature.shape[-1])[0]

    flat_feature = np.mean(spatial_feature_map, axis=0)

    return flat_feature, spatial_feature_map

def word_for_id(integer, tokenizer_loaded):
    for word, index in tokenizer_loaded.word_index.items():
        if index == integer:
            return word
    return None

def generate_caption_and_attention_for_app(model_loaded, feature_extractor_model_loaded, image_np_array, tokenizer_loaded, max_length_loaded, D_CNN_features_loaded, num_image_regions_loaded):
    flat_img_feature, spatial_img_feature = preprocess_image_for_attention(
        image_np_array, feature_extractor_model_loaded
    )
    flat_img_feature = np.expand_dims(flat_img_feature, axis=0)
    spatial_img_feature = np.expand_dims(spatial_img_feature, axis=0)

    in_text = 'startseq'
    sequence = tokenizer_loaded.texts_to_sequences([in_text])[0]
    sequence = pad_sequences([sequence], maxlen=max_length_loaded, padding='post')[0]

    initial_h_layer = model_loaded.get_layer('initial_lstm_h_state')
    initial_c_layer = model_loaded.get_layer('initial_lstm_c_state')

    h_state = initial_h_layer(flat_img_feature)
    c_state = initial_c_layer(flat_img_feature)

    embedding_layer_inference = model_loaded.get_layer('caption_embedding')
    lstm_decoder_inference = model_loaded.get_layer('lstm_decoder')
    attention_layer_inference = model_loaded.get_layer('attention_layer')
    dropout_1_inference = model_loaded.get_layer('dropout_1')
    dense_output_1_inference = model_loaded.get_layer('dense_output_1')
    dropout_2_inference = model_loaded.get_layer('dropout_2')
    output_dense_inference = model_loaded.get_layer('output_word_prediction')

    attention_plot = []
    caption_list = []

    for i in range(max_length_loaded):
        current_sequence_input = np.expand_dims(sequence, axis=0)
        current_embedding = embedding_layer_inference(current_sequence_input)
        lstm_output_inf, h_state, c_state = lstm_decoder_inference(
            current_embedding, initial_state=[h_state, c_state]
        )
        context_vector, weights = attention_layer_inference([spatial_img_feature, h_state])
        combined_features_inf = concatenate([h_state, context_vector], axis=-1)

        x = dropout_1_inference(combined_features_inf, training=False)
        x = dense_output_1_inference(x)
        x = dropout_2_inference(x, training=False)
        prediction = output_dense_inference(x)

        predicted_id = tf.argmax(prediction, axis=-1).numpy()[0]
        predicted_word = word_for_id(predicted_id, tokenizer_loaded)

        attention_plot.append(weights.numpy().reshape(7, 7))
        caption_list.append(predicted_word)

        if predicted_word == 'endseq':
            break

        sequence = np.roll(sequence, -1)
        sequence[-1] = predicted_id

    return caption_list, attention_plot


def plot_attention_map_streamlit(image_raw, generated_words_list, attention_plot_list, img_size=224):
    if isinstance(image_raw, bytes):
        img_pil = tf.keras.preprocessing.image.load_img(tf.io.BytesIO(image_raw))
    else:
        img_pil = tf.keras.preprocessing.image.array_to_img(image_raw, scale=False)

    original_img = np.array(img_pil)

    fig = plt.figure(figsize=(16, 16))

    words_to_display = []
    attention_maps_to_display = []

    for i, word in enumerate(generated_words_list):
        if word == 'startseq':
            continue
        elif word == 'endseq':
            break
        else:
            words_to_display.append(word)
            attention_maps_to_display.append(attention_plot_list[i])

    if not words_to_display:
        st.warning("No words to plot attention for (caption might be too short or contains only tokens).")
        plt.close(fig)
        return None

    num_plots = len(words_to_display)
    cols = min(num_plots, 4)
    rows = (num_plots + cols - 1) // cols

    for i in range(num_plots):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_title(words_to_display[i], fontsize=10)
        ax.imshow(original_img)

        attention_map = attention_maps_to_display[i]
        ax.imshow(tf.image.resize(tf.expand_dims(attention_map, -1), (original_img.shape[0], original_img.shape[1])).numpy().squeeze(),
                  cmap='hot', alpha=0.6)
        ax.axis('off')
    plt.tight_layout()
    return fig


# --- 4. Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="Image Captioning with Attention")

st.title("Image Captioning with Attention")
st.markdown("Upload an image and the model will generate a descriptive caption, highlighting what it 'looks' at for each word.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    st.image(image_bytes, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    if st.button("Generate Caption"):
        st.subheader("Generated Caption:")
        with st.spinner("Generating caption and attention map..."):
            generated_caption_words, attention_weights_list = generate_caption_and_attention_for_app(
                caption_model, feature_extractor_model, image_bytes, tokenizer, max_length, D_CNN_features, num_image_regions
            )
            
            clean_caption = ' '.join([
                word for word in generated_caption_words if word not in ['startseq', 'endseq']
            ])
            st.write(f"**{clean_caption}**")

            st.subheader("Attention Visualization:")
            attention_fig = plot_attention_map_streamlit(
                image_bytes, generated_caption_words, attention_weights_list
            )
            if attention_fig:
                st.pyplot(attention_fig)
            plt.close(attention_fig)
