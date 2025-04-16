import os
import re
import PyPDF2
import textstat
import streamlit as st
from openai import OpenAI

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
    lines = [
        re.sub(r'\s+', ' ', re.sub(r'[^\x00-\x7F]+', ' ', line.strip()))
        for line in text.splitlines() if line.strip()
    ]
    return lines

def simplifytext(text, client, patientcontext=None):
    prompt = (
    f"Patient Context:\n{patientcontext}\n\n"
    f"Medical Instructions:\n{text}\n\n"
    "Please simplify and reformat the discharge instructions by following these steps in order:\n"
    "Rewrite using simple, everyday language that someone with limited medical background can easily understand.\n"
    "Preserve all original section headings and retain every essential detail.\n"
    "Paraphrase to about a 5th–6th grade reading level, without outputting any readability metrics.\n"
    "Convert into a concise, step‑by‑step numbered list, each step beginning with an action verb (e.g., “Take,” “Schedule,” “Monitor”).\n"
    "Use active voice and second‑person (“you”), keeping sentences under 10 words each.\n"
    "Adapt phrasing to be culturally appropriate for the patient’s demographic (if provided).\n"
    "Optionally format as bullet points under each heading if that improves clarity.\n"
    "Ensure the final text would score between 80 and 90 on the Flesch Reading Ease scale—but do not output the score.\n"
)

    if prompt in llmcache:
        return llmcache[prompt]
    try:
        response = client.chat.completions.create(
            model="openai/gpt-3.5-turbo",
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
    return textstat.flesch_reading_ease(simplifiedtext)

st.title("Discharge Instruction Simplifier")
uploadfile = st.file_uploader("Upload Discharge Instructions", type=["txt", "pdf"])
if not uploadfile:
    st.stop()
lines = loadandpreprocess(uploadfile)
if not lines:
    st.warning("No valid data found in the file.")
    st.stop()
original_text = " ".join(lines)
with st.spinner("Initializing OpenRouter client..."):
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=st.secrets["OPENROUTER_API_KEY"])
with st.spinner("Simplifying text without context..."):
    simplified_no_context = simplifytext(original_text, client, patientcontext=None)
    st.subheader("Original Discharge Instructions")
    for paragraph in lines:
        st.write(paragraph)
        st.write("") 
    patient_context_input = st.text_input("Enter patient context (optional):")
    if st.button("Simplify with Patient Context"):
        with st.spinner("Re-simplifying with clinician's context..."):
            simplified_with_context = simplifytext(
                original_text,
                client,
                patientcontext=patient_context_input
            )
        st.subheader("Simplified Text (With Context)")
        st.write(simplified_with_context)
        context_score = evaluatereadability(simplified_with_context)
        st.subheader("Readability Score (Flesch Reading Ease)")
        st.write(context_score)
    else:
        st.info("No context applied yet. Above is the default ‘no context’ simplified text.")
        st.subheader("Current Simplified Text (No Context)")
        st.write(simplified_no_context)
        no_context_score = evaluatereadability(simplified_no_context)
        st.subheader("Readability Score (Flesch Reading Ease)")
        st.write(no_context_score)
