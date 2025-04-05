import os
import PyPDF2
import nltk
import re
import textstat
import nltk
import streamlit as st
from openai import OpenAI

# Ensure necessary NLTK data is downloaded
nltk_packages = ['punkt', 'punkt_tab']
for package in nltk_packages:
    try:
        nltk.data.find(package)
    except Exception as e:
        print(f"Error finding {package} data: {e}")
        nltk.download(package)

def loadandpreprocess(uploadfile):
    """
    Loads text data and performs more sophisticated preprocessing.
    """
    _, ext = os.path.splitext(uploadfile.name)
    if ext.lower() == ".pdf":
        try:
            reader = PyPDF2.PdfReader(uploadfile)
            text = "".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            text = ""
    else:
        try:
            text = uploadfile.read().decode("utf-8")
        except Exception as e:
            st.error(f"Error decoding text file: {e}")
            text = ""
    sections = {}
    currentsection = "General"
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if re.match(r"^(Final Diagnoses|Brief Hospital Course|Discharge Medications|Discharge Instructions|Follow-up Appointments|Signs and Symptoms to Watch For|When to Seek Medical Attention|Contact Information|Notes):", line, re.IGNORECASE):
            currentsection = line.split(":")[0].strip()
            sections.setdefault(currentsection, [])
        else:
            sections.setdefault(currentsection, []).append(line)
    processedsections = {}
    for sectionname, lines in sections.items():
        processedlines = [
            re.sub(r'\s+', ' ', re.sub(r'[^\x00-\x7F]+', ' ', line.strip()))
            for line in lines if line.strip()
        ]
        processedsections[sectionname] = " ".join(processedlines)
    return processedsections

def simplifytext(textsections, client, patientcontext=None):
    """
    Simplifies medical text sections using a more detailed prompt.
    """
    simplifiedsections = {}
    for sectionname, text in textsections.items():
        prompt = (
            f"Patient Context: {patientcontext}\n\n"
            f"Medical Section Title: {sectionname}\n"
            f"Original Medical Text: {text}\n\n"
            f"Simplify the above medical text for a patient with limited medical knowledge. "
            f"Use clear, plain language and avoid medical jargon. Explain any necessary medical terms simply. "
            f"Retain all essential information, including specific instructions, medication names, dosages, frequencies, and follow-up details. "
            f"Structure the information logically and use formatting like bullet points or numbered lists where appropriate for readability. "
            f"Focus on what the patient needs to know and do. If the section is 'Discharge Medications', clearly list each medication with its purpose (if discernible from the text), dosage, and how often to take it. "
            f"If the section is 'Follow-up Appointments', clearly state the type of appointment, date, time, and contact information if provided. "
            f"If the section is 'Signs and Symptoms to Watch For' or 'When to Seek Medical Attention', use simple terms to describe what the patient should look out for and when they should seek help.\n\n"
            f"Simplified Patient-Friendly Text:"
        )
        try:
            response = client.chat.completions.create(
                model="google/gemini-2.0-flash-thinking-exp:free",
                messages=[{"role": "user", "content": prompt}],
            )
            simplifiedsections[sectionname] = response.choices[0].message.content
        except Exception as e:
            simplifiedsections[sectionname] = f"[OpenRouter Error in {sectionname}] {e}"
    return simplified_sections

def extractkeyinfo(simplifiedsections):
    """
    Extracts key information from the simplified text sections.
    """
    keyinfo = []
    keywords = ['follow', 'call', 'take', 'return', 'appointment', 'contact', 'schedule', 'medication']
    for sectionname, simplifiedtext in simplifiedsections.items():
        sentences = nltk.sent_tokenize(simplifiedtext)
        for sent in sentences:
            if any(keyword in sent.lower() for keyword in keywords):
                keyinfo.append(f"[{sectionname}] {sent}")
    return keyinfo

def evaluatereadability(simplified_sections):
    """
    Evaluates the readability of the simplified text sections.
    """
    readabilityscores = {}
    for sectionname, simplifiedtext in simplifiedsections.items():
        score = textstat.flesch_reading_ease(simplifiedtext)
        readabilityscores[sectionname] = score
    return readabilityscores

# Streamlit application
st.title("Discharge Instruction Simplifier")
uploadfile = st.file_uploader("Upload Discharge Instructions", type=["txt", "pdf"])

if uploadfile is not None:
    data = loadandpreprocess(uploadfile)
    if data:
        st.subheader("Original Text Sections")
        for sectionname, text in data.items():
            st.write(f"**{sectionname}:** {text}")

        with st.spinner("Initializing OpenRouter client..."):
            client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=st.secrets["OPENROUTER_API_KEY"])

        patientcontext = st.text_input("Enter patient context (optional):")

        with st.spinner("Simplifying text..."):
            simplifieddata = simplifytext(data, client, patientcontext=patientcontext)

        st.subheader("Simplified Text Sections")
        for sectionname, text in simplifieddata.items():
            st.write(f"**{sectionname}:** {text}")

        keyinfo = extractkeyinfo(simplifieddata)
        st.subheader("Extracted Key Information")
        st.write(keyinfo)

        readability = evaluatereadability(simplifieddata)
        st.subheader("Readability Scores (Flesch Reading Ease) per Section")
        st.write(readability)

    else:
        st.warning("No valid data found in the file.")
else:
    st.info("Please upload a discharge instructions file.")
