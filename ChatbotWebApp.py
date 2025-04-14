import os
import re
import nltk
import PyPDF2
import textstat
import streamlit as st
from openai import OpenAI

# Ensure required NLTK packages are available
for package in ['punkt', 'punkt_tab']:
    try:
        nltk.data.find(package)
    except Exception as e:
        print(f"Error finding {package} data: {e}")
        nltk.download(package)

llmcache = {}

def loadandpreprocess(uploadfile):
    _, ext = os.path.splitext(uploadfile.name)
    text = ""
    if ext.lower() == ".pdf":
        try:
            reader = PyPDF2.PdfReader(uploadfile)
            text = "".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
    elif ext.lower() == ".txt":
        try:
            text = uploadfile.read().decode("utf-8")
        except Exception as e:
            st.error(f"Error reading TXT file: {e}")
    else:
        st.error("Unsupported file format. Please upload a PDF or TXT file.")
    
    return [
        re.sub(r'\s+', ' ', re.sub(r'[^\x00-\x7F]+', ' ', line.strip()))
        for line in text.splitlines() if line.strip()
    ]

def simplifytext(text, client, patientcontext=None):
    # DO NOT change the prompt below
    prompt = (
        f"Patient Context:\n{patientcontext}\n\n"
        f"Medical Instructions:\n{text}\n\n"
        "Use simple, clear language that someone with limited medical knowledge can easily understand.\n\n"
        "Convert the following discharge instructions into plain, patient-friendly language, ensuring accuracy with respect to the MTSamples discharge summary. "
        "Retain all essential details while reformulating the text so that it achieves a Flesch Reading Ease score between 80 and 90. Dont output Flesch Reading Ease score check "
        "final simplified text should be focused on list of tasks, follow-ups, and their importance from the discharge instructions."
    )
    if prompt in llmcache:
        return llmcache[prompt]
    try:
        response = client.chat.completions.create(
            model="openrouter/auto",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            top_p=1
        )
        result = response.choices[0].message.content
        llmcache[prompt] = result
        return result
    except Exception as e:
        return f"[OpenRouter Error] {e}"

def evaluatereadability(simplifiedtext):
    score = textstat.flesch_reading_ease(simplifiedtext)
    return score

##########################
# Streamlit App Interface
##########################

st.title("Discharge Instruction Simplifier")

# 1. Let the user choose their view BEFORE uploading a file.
view_type = st.radio("Choose your view:", ["Patient", "Clinician"], index=0)

# 2. File uploader appears next.
uploadfile = st.file_uploader("Upload Discharge Instructions", type=["txt", "pdf"])

# 3. If file has been uploaded, proceed to text processing.
if uploadfile is not None:
    data = loadandpreprocess(uploadfile)
    if data:
        originaltext = " ".join(data)

        with st.spinner("Initializing OpenRouter client..."):
            # The OpenRouter client is used for completions.
            client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=st.secrets["OPENROUTER_API_KEY"])

        # Optional patient context input.
        patientcontext = st.text_input("Enter patient context (optional):")

        with st.spinner("Simplifying the text..."):
            simplifiedtext = simplifytext(originaltext, client, patientcontext=patientcontext)

        # 4. Display results depending on view type.
        if view_type == "Patient":
            st.subheader("Simplified Text")
            st.write(simplifiedtext)

            readability = evaluatereadability(simplifiedtext)
            st.subheader("Readability Score (Flesch Reading Ease)")
            st.write(readability)

        else:  # Clinician
            st.subheader("Original Discharge Instructions")
            st.write(originaltext)

            if patientcontext:
                st.subheader("Patient Context Provided")
                st.write(patientcontext)

            st.subheader("Simplified Text (Patient-Friendly)")
            st.write(simplifiedtext)

            readability = evaluatereadability(simplifiedtext)
            st.subheader("Readability Score (Flesch Reading Ease)")
            st.write(readability)
    else:
        st.warning("No valid data found in the file.")
else:
    st.info("Please upload a discharge instructions file (PDF or TXT).")
