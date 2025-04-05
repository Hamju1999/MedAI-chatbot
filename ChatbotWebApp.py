import os
import PyPDF2
import nltk
import re
import textstat
import nltk
import streamlit as st
from openai import OpenAI

for package in ['punkt', 'punkt_tab']:
    try:
        nltk.data.find(package)
    except Exception as e:
        print(f"Error finding {package} data: {e}")
        nltk.download(package)

def loadandpreprocess(uploadfile):
    _, ext = os.path.splitext(uploadfile.name)
    if ext.lower() == ".pdf":
        try:
            reader = PyPDF2.PdfReader(uploadfile)
            text = "".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            text = ""
    else:
        text = uploadfile.read().decode("utf-8")    
    return [
        re.sub(r'\s+', ' ', re.sub(r'[^\x00-\x7F]+', ' ', line.strip()))
        for line in text.splitlines() if line.strip()
    ]

def simplifytext(text, client, patientcontext=None):
    prompt = f"Patient context: {patientcontext}\nMedical Instructions: {text}" if patientcontext else text
    message = (
        "Simplify the following medical instructions into clear, patient-friendly language. "
        "Retain the essential details but use plain language and structure the information for easy reading:\n\n" 
        + prompt
    )
    try:
        response = client.chat.completions.create(
            model="google/gemini-2.0-flash-thinking-exp:free",
            messages=[{"role": "user", "content": message}],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[OpenRouter Error] {e}"

def extractkeyinfo(simplifiedtext):
    sentences = nltk.sent_tokenize(simplifiedtext)
    keywords = ['follow', 'call', 'take', 'return', 'appointment', 'contact', 'schedule', 'medication']
    keyphrases = [sent for sent in sentences if any(keyword in sent.lower() for keyword in keywords)]
    return keyphrases

def evaluatereadability(simplifiedtext):
    score = textstat.flesch_reading_ease(simplifiedtext)
    return score

st.title("Discharge Instruction")
uploadfile = st.file_uploader("Upload Discharge Instructions", type=["txt", "pdf"])

if uploadfile is not None:
    data = loadandpreprocess(uploadfile)
    if data:
        originaltext = " ".join(data)
        st.subheader("Original Text")
        st.write(originaltext)
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=st.secrets["OPENROUTER_API_KEY"])
        patientcontext = st.text_input("Enter patient context (optional):")
        with st.spinner("Simplifying text..."):
            simplifiedtext = simplifytext(originaltext, client, patientcontext=patientcontext)
        st.subheader("Simplified Text")
        st.write(simplifiedtext)
        keyinfo = extractkeyinfo(simplifiedtext)
        st.subheader("Extracted Key Information")
        st.write(keyinfo)
        readability = evaluatereadability(simplifiedtext)
        st.subheader("Readability Score (Flesch Reading Ease)")
        st.write(readability)
    else:
        st.warning("No valid data found in the file.")
else:
    st.info("Please upload a discharge instructions file.")
