# app.py

import os
import re
import requests

import streamlit as st

# optional readability scoring
try:
    import textstat
except ImportError:
    textstat = None

# --- UI setup ---
st.set_page_config(page_title="Discharge Summary Simplifier", layout="wide")
st.title("Discharge Summary Simplifier")
st.subheader("Transforming Patient Understanding with AI")
st.write("Upload a hospital discharge summary and let the AI simplify it into clear instructions.")

# --- API key handling ---
api_key = os.getenv("OPENROUTER_API_KEY", "")
if not api_key:
    api_key = st.text_input("Enter your OpenRouter API Key", type="password")
if not api_key:
    st.info("🔑 Please provide your OpenRouter API key.")
    st.stop()

# --- File uploader ---
uploaded_file = st.file_uploader(
    "Upload Discharge Summary (TXT or PDF)",
    type=["txt", "pdf"],
    help="Accepted formats: .txt or .pdf"
)
if not uploaded_file:
    st.stop()

# --- Options ---
col1, col2 = st.columns(2)
with col1:
    reading_level = st.slider(
        "Target Reading Level (US Grade)",
        min_value=3, max_value=12, value=6
    )
with col2:
    language = st.selectbox(
        "Output Language",
        ["English", "Spanish", "Chinese", "French", "German"],
        index=0
    )

# --- Text extraction ---
def extract_text_from_file(file) -> str:
    """Read text from uploaded TXT or PDF."""
    if file.type == "application/pdf":
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            st.error("PDF support requires PyPDF2. Please `pip install PyPDF2`.")
            return ""
        # Attempt direct reading; fallback to bytes IO
        try:
            reader = PdfReader(file)
        except Exception:
            import io
            reader = PdfReader(io.BytesIO(file.read()))
        text = ""
        for page in reader.pages:
            text += (page.extract_text() or "") + "\n"
        return text
    else:
        data = file.read()
        for encoding in ("utf-8", "latin-1"):
            try:
                return data.decode(encoding)
            except Exception:
                continue
        return ""

discharge_text = extract_text_from_file(uploaded_file).strip()
if not discharge_text:
    st.error("❌ Could not extract any text. Please upload a valid TXT or PDF.")
    st.stop()

# --- LLM call ---
def summarize_discharge(text: str, reading_lvl: int, lang: str) -> dict:
    prompt = (
        f"The following is a hospital discharge summary.\n"
        f"Simplify it to a {reading_lvl}th-grade reading level in {lang}.\n"
        f"Break into sections with these headings:\n"
        "- **Simplified Instructions:** bullet‑points\n"
        "- **Importance:** why each instruction matters\n"
        "- **Follow-Up Appointments or Tasks:** tasks/visits\n"
        "- **Medications:** with simple dosing notes\n"
        "- **Precautions:** warning signs, activities to avoid\n"
        "- **References:** brief reasons/explanations\n\n"
        f"Now simplify:\n\"\"\"{text}\"\"\""
    )
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek/deepseek-r1",
        "messages": [{"role": "user", "content": prompt}]
    }
    resp = requests.post(url, headers=headers, json=payload)
    return {
        "status": resp.status_code,
        "raw_json": resp.json() if resp.headers.get("content-type","").startswith("application/json") else resp.text
    }

# --- Main action ---
if st.button("Simplify Discharge Instructions"):

    with st.spinner("🧠 Summarizing…"):
        api_resp = summarize_discharge(discharge_text, reading_level, language)

    # show raw JSON
    st.subheader("🔍 Raw API response")
    st.json(api_resp["raw_json"])

    if api_resp["status"] != 200:
        st.error(f"API returned status {api_resp['status']}.")
        st.stop()

    # pull out the content
    choices = api_resp["raw_json"].get("choices", [])
    simplified_text = ""
    if choices:
        simplified_text = choices[0].get("message", {}).get("content", "").strip()

    if not simplified_text:
        st.error("No content returned by the LLM.")
        st.stop()

    # show raw LLM text
    st.subheader("📝 Raw LLM Output")
    st.text_area("raw_output", simplified_text, height=300)

    # render markdown directly
    st.markdown("---")
    st.subheader("📄 Formatted Simplified Summary")
    st.markdown(simplified_text)

    # optional: calculate readability
    st.markdown("---")
    st.subheader("🔢 Readability")
    if textstat:
        score = textstat.flesch_kincaid_grade(simplified_text)
        st.write(f"Flesch‑Kincaid Grade Level: **{score:.1f}**")
    else:
        st.write("Install `textstat` for readability metrics.")

    # optional: split into sections via regex
    st.markdown("---")
    st.subheader("🔀 Parsed Sections")
    header_re = re.compile(r"^\*{0,2}(.+?)\*{0,2}:?$")
    sections = {
        "Simplified Instructions": [],
        "Importance": [],
        "Follow-Up Appointments or Tasks": [],
        "Medications": [],
        "Precautions": [],
        "References": []
    }
    current = None
    for line in simplified_text.splitlines():
        m = header_re.match(line.strip())
        if m:
            key = m.group(1).strip()
            if key in sections:
                current = key
                continue
        if current:
            sections[current].append(line)

    for sec, lines in sections.items():
        if lines:
            st.markdown(f"**{sec}**")
            st.write("\n".join(lines))
            st.markdown("")  # spacing
