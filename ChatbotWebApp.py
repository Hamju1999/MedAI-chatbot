import os
import PyPDF2
import nltk
import re
import textstat
import json
import requests
import streamlit as st
from openai import OpenAI

# Ensure required NLTK data packages are available
for package in ['punkt', 'punkt_tab']:
    try:
        nltk.data.find(package)
    except Exception as e:
        print(f"Error finding {package} data: {e}")
        nltk.download(package)

llmcache = {}

def loadandpreprocess(uploadfile):
    """Read and preprocess only PDFs."""
    _, ext = os.path.splitext(uploadfile.name)
    text = ""
    if ext.lower() == ".pdf":
        try:
            reader = PyPDF2.PdfReader(uploadfile)
            text = "".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
    else:
        st.warning("Only PDF files are supported for simplification.")
    # Clean and filter text lines
    return [
        re.sub(r'\s+', ' ', re.sub(r'[^\x00-\x7F]+', ' ', line.strip()))
        for line in text.splitlines() if line.strip()
    ]

def simplifytext(text, client, patientcontext=None, trainingdata=None):
    """
    Build a prompt that includes a very limited training context from JSON to instruct the LLM without
    outputting the JSON itself, and truncate the PDF text to a much smaller excerpt.
    """
    # Build a very limited training context
    training_context = ""
    max_samples = 3  # Use only up to 3 training examples
    if trainingdata and "discharge_samples" in trainingdata:
        training_context += "Learn from these examples of simplifying complex medical instructions:\n\n"
        for sample in trainingdata["discharge_samples"][:max_samples]:
            if "structure" in sample:
                for section in sample["structure"]:
                    if "heading_analysis" in section:
                        ha = section["heading_analysis"]
                        if (ha.get("complexity_level", "").lower() == "complex" and 
                            ha.get("original_heading") and ha.get("simplified_explanation")):
                            training_context += (
                                f"Example: Convert '{ha.get('original_heading')}' to '{ha.get('simplified_explanation')}'.\n"
                            )
                    if "paragraphs" in section:
                        for paragraph in section["paragraphs"]:
                            if "paragraph_analysis" in paragraph:
                                pa = paragraph["paragraph_analysis"]
                                if (pa.get("complexity_level", "").lower() == "complex" and 
                                    pa.get("original_paragraph") and pa.get("simplified_explanation")):
                                    training_context += (
                                        f"Example: Convert '{pa.get('original_paragraph')}' to '{pa.get('simplified_explanation')}'.\n"
                                    )
                            if "words_analysis" in paragraph:
                                for wordanalysis in paragraph["words_analysis"]:
                                    if (wordanalysis.get("complexity_level", "").lower() == "complex" and 
                                        wordanalysis.get("word") and wordanalysis.get("simplified_explanation")):
                                        training_context += (
                                            f"Example: Convert '{wordanalysis.get('word')}' to '{wordanalysis.get('simplified_explanation')}'.\n"
                                        )
        # Limit the training context length further, if needed.
        max_training_length = 1000  # characters
        if len(training_context) > max_training_length:
            training_context = training_context[:max_training_length] + "\n[Truncated training context]\n"

    # Aggressively truncate the PDF text
    max_text_length = 1000  # characters
    if len(text) > max_text_length:
        text = text[:max_text_length] + "\n[Truncated]"

    # Construct the final prompt
    prompt = training_context
    if patientcontext:
        prompt += f"Patient Context:\n{patientcontext}\n\n"
    prompt += (
        f"Medical Instructions:\n{text}\n\n"
        "Use simple, clear language that someone with limited medical knowledge can easily understand.\n\n"
        "Convert the above discharge instructions into plain, patient-friendly language. "
        "Retain all essential details and focus on a list of tasks, follow-ups, and their importance."
    )
    
    # Log the prompt length for debugging purposes
    st.write("Prompt length (characters):", len(prompt))

    if prompt in llmcache:
        return llmcache[prompt]
    
    try:
        response = client.chat.completions.create(
            model="google/gemini-2.0-flash-thinking-exp:free",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            top_p=1
        )
        # Check if the response has valid choices
        if response is None or not hasattr(response, "choices") or not response.choices:
            return f"[OpenRouter Error] No valid response choices received. (Prompt length: {len(prompt)} characters)"
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

# Streamlit interface setup
st.title("Discharge Instruction Simplifier")
uploadfile = st.file_uploader("Upload Discharge Instructions (PDF only)", type=["pdf"])

# Load training data from GitHub without outputting it to the user
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
            client = OpenAI(base_url="https://openrouter.ai/api/v1", 
                            api_key=st.secrets["OPENROUTER_API_KEY"])
        # Optionally, capture patient context (left empty if not provided)
        patientcontext = st.text_input("Enter patient context (optional):")
        with st.spinner("Simplifying text..."):
            simplifiedtext = simplifytext(originaltext, client, patientcontext=patientcontext, trainingdata=trainingdata)
        st.subheader("Simplified Text")
        st.write(simplifiedtext)
        readability = evaluatereadability(simplifiedtext)
        st.subheader("Readability Score (Flesch Reading Ease)")
        st.write(readability)
    else:
        st.warning("No valid text data found in the file.")
else:
    st.info("Please upload a PDF file containing discharge instructions.")
