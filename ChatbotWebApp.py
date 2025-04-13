import os
import PyPDF2
import nltk
import re
import textstat
import nltk
import json
import requests
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

def simplifytext(text, client, patientcontext=None, trainingdata=None):
    promptexamples = ""
    if trainingdata and "discharge_samples" in trainingdata:
        promptexamples += ("Here are some examples of complex medical terms, headings, and paragraphs with "
                           "their simplified explanations to learn from:\n\n")
        for sample in trainingdata["discharge_samples"]:
            sample_url = sample.get("sample_url")
            if sample_url:
                promptexamples += f"Sample URL: {sample_url}\n\n"
            if "structure" in sample:
                for section in sample["structure"]:
                    # Check the heading_analysis for the section heading
                    if "heading_analysis" in section:
                        ha = section["heading_analysis"]
                        # Using original_heading from the analysis if available (could also use section["heading"])
                        if (ha.get("complexity_level", "").lower() == "complex" and 
                            ha.get("original_heading") and ha.get("simplified_explanation")):
                            promptexamples += (f"Complex Heading: {ha.get('original_heading')}\n"
                                               f"Simplified: {ha.get('simplified_explanation')}\n\n")
                    # Process each paragraph within the section
                    if "paragraphs" in section:
                        for paragraph in section["paragraphs"]:
                            # Check paragraph_analysis first
                            if "paragraph_analysis" in paragraph:
                                pa = paragraph["paragraph_analysis"]
                                if (pa.get("complexity_level", "").lower() == "complex" and 
                                    pa.get("original_paragraph") and pa.get("simplified_explanation")):
                                    promptexamples += (f"Complex Paragraph: {pa.get('original_paragraph')}\n"
                                                       f"Simplified: {pa.get('simplified_explanation')}\n\n")
                            # Then process words_analysis if present
                            if "words_analysis" in paragraph:
                                for wordanalysis in paragraph["words_analysis"]:
                                    word = wordanalysis.get("word")
                                    simplifiedexplanation = wordanalysis.get("simplified_explanation")
                                    complexitylevel = wordanalysis.get("complexity_level")
                                    if (complexitylevel and complexitylevel.lower() == "complex" and 
                                        word and simplifiedexplanation):
                                        promptexamples += (f"Complex Term: {word}\n"
                                                           f"Simplified: {simplifiedexplanation}\n\n")
    return promptexamples

    prompt = (
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
githuburl = "https://github.com/Hamju1999/MedAI-chatbot/blob/master/train.json"
rawgithub = githuburl.replace("github.com", "raw.githubusercontent.com").replace("/blob", "")
trainingdata = None

with st.spinner(f"Loading training data from: {rawgithub}"):
    try:
        response = requests.get(rawgithub)
        response.raise_for_status()  
        trainingdata = response.json()
        st.success("Training data loaded successfully from GitHub!")
    except requests.exceptions.RequestException as e:
        st.error(f"Error loading training data from GitHub: {e}")
    except json.JSONDecodeError:
        st.error("Error: Invalid JSON file in GitHub.")
    except Exception as e:
        st.error(f"An unexpected error occurred while loading training data: {e}")

if uploadfile is not None:
    data = loadandpreprocess(uploadfile)
    if data:
        originaltext = " ".join(data)
        with st.spinner("Initializing OpenRouter client..."):
            client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=st.secrets["OPENROUTER_API_KEY"])
        patientcontext = st.text_input("Enter patient context (optional):")
        with st.spinner("Simplifying text..."):
            simplifiedtext = simplifytext(originaltext, client, patientcontext=patientcontext, trainingdata=trainingdata)
        st.subheader("Simplified Text")
        st.write(simplifiedtext)
        readability = evaluatereadability(simplifiedtext)
        st.subheader("Readability Score (Flesch Reading Ease)")
        st.write(readability)
    else:
        st.warning("No valid data found in the file.")
else:
    st.info("Please upload a discharge instructions file.")
