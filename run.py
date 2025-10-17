import streamlit as st
from PIL import Image
from gtts import gTTS
import os
import tempfile
import io
import random

# Audio & Speech
import whisper
import speech_recognition as sr
from pydub import AudioSegment

# Image & DeepFace
from deepface import DeepFace
import numpy as np
from rembg import remove

# Text Analysis
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import spacy
from spacy import displacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# -------------------------------
# Main App
# -------------------------------
st.title("🧠 Unstructured Data Analysis")

tab1, tab2, tab3 = st.tabs(["🖼 Image Analysis", "🎧 Audio Analysis", "📝 Text Analysis"])

# -------------------------------
# Tab 1: Image Analysis
# -------------------------------
with tab1:
    st.header("🖼 Image Analysis with DeepFace")

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        img_array = np.array(image)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("Detect Face"):
                try:
                    detection = DeepFace.detectFace(img_array, enforce_detection=True)
                    st.success("✅ Face detected!")
                    st.image(detection, caption="Detected Face", use_column_width=True)
                except Exception as e:
                    st.error(f"Face detection failed: {e}")

        with col2:
            if st.button("Detect Age & Gender"):
                try:
                    analysis = DeepFace.analyze(img_path=img_array, actions=['age', 'gender'], enforce_detection=True)
                    predicted_age = analysis[0]['age']
                    predicted_gender = analysis[0]['dominant_gender']
                    st.success("✅ Age & Gender detected!")
                    st.write(f"*Predicted Age:* {predicted_age}")
                    st.write(f"*Predicted Gender:* {predicted_gender}")
                except Exception as e:
                    st.error(f"Age/Gender detection failed: {e}")

        with col3:
            if st.button("Detect Emotion"):
                try:
                    analysis = DeepFace.analyze(img_path=img_array, actions=['emotion'], enforce_detection=True)
                    predicted_emotion = analysis[0]['dominant_emotion']
                    st.success("✅ Emotion detected!")
                    st.write(f"*Predicted Emotion:* {predicted_emotion}")
                except Exception as e:
                    st.error(f"Emotion detection failed: {e}")

        with col4:
            try:
                output_image = remove(image)
                st.image(output_image, caption="Background Removed", width=300)
            except Exception as e:
                st.error(f"BG removal failed: {e}")

# -------------------------------
# Tab 2: Audio Analysis
# -------------------------------
with tab2:
    st.header("🎧 Audio Analysis")

    # Text to Speech
    st.subheader("🗣 Text to Speech")
    text_input = st.text_area("Enter text to convert to speech:")

    if st.button("Convert to Audio"):
        if text_input.strip():
            tts = gTTS(text_input, lang='en')
            tts.save("output.mp3")
            audio_file = open("output.mp3", "rb")
            st.audio(audio_file.read(), format='audio/mp3')
            st.success("✅ Conversion complete!")
        else:
            st.warning("Please enter some text.")

    # Speech to Text
    st.subheader("🗣 Speech to Text")
    os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"  # Ensure FFmpeg path

    @st.cache_resource(show_spinner=False)
    def load_whisper_model(model_name="base"):
        return whisper.load_model(model_name)

    model = load_whisper_model("base")
    st.success("✅ Whisper model loaded")

    uploaded_audio = st.file_uploader("Upload an audio file (wav/mp3/m4a)", type=["wav", "mp3", "m4a"])

    if uploaded_audio:
        suffix = "." + uploaded_audio.name.split(".")[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_file_path = tmp.name
            tmp.write(uploaded_audio.read())

        st.audio(temp_file_path)

        if st.button("Transcribe Audio"):
            with st.spinner("Transcribing..."):
                if os.path.exists(temp_file_path):
                    try:
                        result = model.transcribe(temp_file_path)
                        st.success("✅ Transcription complete!")
                        st.subheader("Transcribed Text")
                        st.write(result["text"])
                    except Exception as e:
                        st.error(f"Error during transcription: {e}")
                else:
                    st.error("❌ Temporary audio file not found!")

# -------------------------------
# Tab 3: Text Analysis
# -------------------------------
with tab3:
    st.header("📝 Text Analysis")

    # Sample stories
    stories = [
        "In a remote kingdom nestled between jagged mountains and endless forests...",
        "During the bustling era of the 1920s, Detective Samuel Hart navigated...",
        "On a distant exoplanet, Captain Rhea led a team of explorers...",
        "In the neon-lit heart of Tokyo, young coder Akira toiled over lines of code...",
        "Deep in the Amazon rainforest, a team of scientists embarked on an expedition..."
    ]

    if "text_area" not in st.session_state:
        st.session_state.text_area = ""

    if st.button("🎲 Random Story"):
        st.session_state.text_area = random.choice(stories)

    st.session_state.text_area = st.text_area(
        "Paste or modify your text here:", 
        value=st.session_state.text_area, 
        height=250
    )

    if st.button("Analyze Text 🚀"):
        text = st.session_state.text_area.strip()
        if text:
            # POS tagging with TextBlob
            blob = TextBlob(text)
            words_and_tags = blob.tags

            nouns = [w for w,t in words_and_tags if t.startswith("NN")]
            verbs = [w for w,t in words_and_tags if t.startswith("VB")]
            adjectives = [w for w,t in words_and_tags if t.startswith("JJ")]
            adverbs = [w for w,t in words_and_tags if t.startswith("RB")]

            # WordCloud generator
            def make_wordcloud(words, color):
                if not words:
                    return None
                wc = WordCloud(width=500, height=400, background_color='black', colormap=color).generate(" ".join(words))
                fig, ax = plt.subplots()
                ax.imshow(wc, interpolation='bilinear')
                ax.axis("off")
                return fig

            col1, col2 = st.columns(2)
            col3, col4 = st.columns(2)

            with col1:
                st.subheader("🧠 Nouns")
                fig = make_wordcloud(nouns, "plasma")
                if fig: st.pyplot(fig)
            with col2:
                st.subheader("⚡ Verbs")
                fig = make_wordcloud(verbs, "inferno")
                if fig: st.pyplot(fig)
            with col3:
                st.subheader("🎨 Adjectives")
                fig = make_wordcloud(adjectives, "cool")
                if fig: st.pyplot(fig)
            with col4:
                st.subheader("💨 Adverbs")
                fig = make_wordcloud(adverbs, "magma")
                if fig: st.pyplot(fig)

            # Quick POS stats
            st.write({
                "Nouns": len(nouns),
                "Verbs": len(verbs),
                "Adjectives": len(adjectives),
                "Adverbs": len(adverbs)
            })

            # spaCy NER
            doc = nlp(text)
            html = displacy.render(doc, style="ent", jupyter=False)
            st.markdown("### 🏷 Detected Entities:", unsafe_allow_html=True)
            st.markdown(html, unsafe_allow_html=True)

            entities = [(ent.text, ent.label_) for ent in doc.ents]
            if entities:
                st.table(entities)
            else:
                st.info("No named entities found.")
        else:
            st.warning("Please paste or select some text first.")