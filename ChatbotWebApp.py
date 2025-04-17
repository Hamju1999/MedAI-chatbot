# Import necessary libraries
import streamlit as st
import requests
import os
try:
    import textstat
except ImportError:
    textstat = None

# --- UI Initialization & Setup ---
st.set_page_config(page_title="Discharge Summary Simplifier", layout="wide")
st.title("Discharge Summary Simplifier")
st.subheader("Transforming Patient Understanding with AI")
st.write("Upload a hospital discharge summary and let the AI simplify it into clear instructions.")

# --- API Key Input ---
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    api_key = st.text_input("Enter your OpenRouter API Key", type="password", help="Needed to call the LLM API for summarization.")
    if not api_key:
        st.info("Please enter your OpenRouter API key to use the summarization service.")
        st.stop()

# --- File Upload ---
uploaded_file = st.file_uploader("Upload Discharge Summary (text or PDF)", type=["txt", "pdf"], help="Upload the patient's discharge instructions in TXT or PDF format.")
if not uploaded_file:
    st.stop()

# --- Summarization Options ---
col1, col2 = st.columns(2)
with col1:
    reading_level = st.slider("Target Reading Level (Grade)", min_value=3, max_value=12, value=6, help="The grade level for the simplified text (lower means simpler).")
with col2:
    language = st.selectbox("Output Language", ["English", "Spanish", "Chinese", "French", "German"], index=0, help="Language for the simplified instructions.")

# --- File Reading & Extraction ---
def extract_text_from_file(file):
    text_content = ""
    if file.type == "application/pdf":
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            st.error("PyPDF2 is required for PDF support. Please install it.")
            return ""
        try:
            pdf_reader = PdfReader(file)
        except Exception as e:
            file.seek(0)
            import io
            pdf_reader = PdfReader(io.BytesIO(file.read()))
        for page in pdf_reader.pages:
            text_content += page.extract_text() + "\n"
    else:
        bytes_data = file.read()
        try:
            text_content = bytes_data.decode("utf-8")
        except:
            text_content = bytes_data.decode("latin-1")
    return text_content

# Extract text from the uploaded file
discharge_text = extract_text_from_file(uploaded_file)
if not discharge_text.strip():
    st.error("Unable to read content from the file. Please upload a valid text or PDF file.")
    st.stop()

# --- LLM Summarization (OpenRouter API) ---
def summarize_discharge(text, reading_lvl=6, lang="English"):
    prompt = (
        f"The following is a hospital discharge summary for a patient.\n"
        f"Your task is to simplify these discharge instructions so that a patient at a {reading_lvl}th-grade reading level can understand them. "
        f"Present the information in the patient's preferred language: {lang}. "
        f"Break down the content into the following sections with clear headings:\n"
        f"**Simplified Instructions:** A bullet-point list of the key instructions the patient needs to follow in simple language.\n"
        f"**Importance:** A brief explanation of how important or critical each item is.\n"
        f"**Follow-Up Appointments or Tasks:** List any follow-up appointments or tasks the patient needs to do.\n"
        f"**Medications:** List any medications the patient needs to take, with simplified instructions if available.\n"
        f"**Precautions:** Any precautions or warning signs the patient should be aware of (e.g., symptoms to watch for, activities to avoid).\n"
        f"**References:** Brief reasons or explanations for the above instructions (why each instruction or medication is important).\n\n"
        f"Please output each section clearly. Use simple language and short sentences.\n"
        f"Now, simplify the following discharge summary:\n\"\"\"{text}\"\"\""
    )
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "openai/gpt-3.5-turbo",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    try:
        response = requests.post(url, headers=headers, json=data)
    except Exception as e:
        st.error(f"Failed to connect to the OpenRouter API: {e}")
        return None
    if response.status_code != 200:
        st.error(f"OpenRouter API error: {response.status_code} - {response.text}")
        return None
    result = response.json()
    simplified_output = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    return simplified_output

# --- Generate Summary and Display ---
if st.button("Simplify Discharge Instructions"):
    with st.spinner("Summarizing the discharge instructions..."):
        simplified_text = summarize_discharge(discharge_text, reading_level, language)
    if simplified_text:
        st.success("Discharge instructions simplified successfully!")
        sections = {
            "Simplified Instructions": "",
            "Importance": "",
            "Follow-Up Appointments or Tasks": "",
            "Medications": "",
            "Precautions": "",
            "References": ""
        }
        current_section = None
        for line in simplified_text.splitlines():
            if not line.strip():
                continue
            header_line = line.strip().strip(":").lower()
            if header_line in [key.lower() for key in sections.keys()]:
                for key in sections.keys():
                    if header_line == key.lower():
                        current_section = key
                        sections[current_section] = ""
                        break
            else:
                if current_section:
                    if sections[current_section] and not line.startswith(('-', '*', '•')):
                        sections[current_section] += "\n"
                    sections[current_section] += line + "\n"
        for section, content in sections.items():
            if content:
                st.subheader(section)
                st.markdown(content.strip())
        st.markdown("---")
        st.subheader("Readability Score")
        if textstat:
            score = textstat.flesch_kincaid_grade(simplified_text)
            st.write(f"Flesch-Kincaid Grade Level: **{score:.1f}**")
        else:
            st.write("Install the `textstat` library to calculate readability scores.")
        st.caption("*(The Flesch-Kincaid Grade Level estimates the U.S. school grade required to understand the text.)*")
