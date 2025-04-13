import os
import PyPDF2
import nltk
import re
import textstat
import nltk
import json
import streamlit as st
from openai import OpenAI

for package in ['punkt', 'punkt_tab']:
    try:
        nltk.data.find(package)
    except Exception as e:
        print(f"Error finding {package} data: {e}")
        nltk.download(package)
llmcache = {}

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

def simplifytext(text, client, patientcontext=None, training_data=None):
    prompt_examples = ""
    if training_data and "discharge_samples" in training_data:
        prompt_examples = "Here are some examples of complex medical terms and their simplified explanations to learn from:\n\n"
        for sample in training_data["discharge_samples"]:
            if "structure" in sample:
                for section in sample["structure"]:
                    if "paragraphs" in section:
                        for paragraph in section["paragraphs"]:
                            if "words_analysis" in paragraph:
                                for word_analysis in paragraph["words_analysis"]:
                                    word = word_analysis.get("word")
                                    simplified_explanation = word_analysis.get("simplified_explanation")
                                    complexity_level = word_analysis.get("complexity_level")
                                    if complexity_level and complexity_level.lower() == "complex" and word and simplified_explanation:
                                        prompt_examples += f"Complex Term: {word}\nSimplified: {simplified_explanation}\n\n"

    prompt = (
        f"{prompt_examples}"
        f"Patient Context:\n{patientcontext}\n\n"
        f"Medical Instructions:\n{text}\n\n"
        "Use simple, clear language that someone with limited medical knowledge can easily understand.\n\n"
        "Convert the following discharge instructions into plain, patient-friendly language, ensuring accuracy with respect to the MTSamples discharge summary. "
        "Retain all essential details while reformulating the text so that it achieves a Flesch Reading Ease score between 80 and 90. Dont output Flesch Reading Ease score check\n\n"
        "final simplified text should be focused on list of tasks, follow-ups, and their importance from the discharge instructions."
    )
    if prompt in llmcache:
        return llmcache[prompt]
    try:
        response = client.chat.completions.create(
            model="google/gemini-2.0-flash-thinking-exp:free",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            top_p=1
        )
        result = response.choices[0].message.content
        llmcache[prompt] = result
        return result
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

st.title("Discharge Instruction Simplifier")
uploadfile = st.file_uploader("Upload Discharge Instructions (txt or pdf)", type=["txt", "pdf"])
training_file = st.file_uploader("Upload Training Data (JSON - optional)", type=["json"])

training_data = None
if training_file is not None:
    try:
        training_data = json.load(training_file)
        st.success("Training data loaded successfully!")
    except json.JSONDecodeError:
        st.error("Error: Invalid JSON file.")
    except Exception as e:
        st.error(f"An error occurred while loading the JSON file: {e}")

if uploadfile is not None:
    data = loadandpreprocess(uploadfile)
    if data:
        originaltext = " ".join(data)
        with st.spinner("Initializing OpenRouter client..."):
            client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=st.secrets["OPENROUTER_API_KEY"])
        patientcontext = st.text_input("Enter patient context (optional):")
        with st.spinner("Simplifying text..."):
            simplifiedtext = simplifytext(originaltext, client, patientcontext=patientcontext, training_data=training_data)
        st.subheader("Simplified Text")
        st.write(simplifiedtext)
        readability = evaluatereadability(simplifiedtext)
        st.subheader("Readability Score (Flesch Reading Ease)")
        st.write(readability)
    else:
        st.warning("No valid data found in the file.")
else:
    st.info("Please upload a discharge instructions file.")
