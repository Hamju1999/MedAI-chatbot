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
    Loads text data, performs preprocessing, and sections the text.
    Returns both the full original text and the processed sections.
    """
    _, ext = os.path.splitext(uploadfile.name)
    fulltext = ""  # Initialize full_text here
    if ext.lower() == ".pdf":
        try:
            reader = PyPDF2.PdfReader(uploadfile)
            textpages = [page.extract_text() or "" for page in reader.pages]
            fulltext = "".join(textpages)
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
    else:
        try:
            full_text = uploadfile.read().decode("utf-8")
        except Exception as e:
            st.error(f"Error decoding text file: {e}")
    sections = {}
    currentsection = "General"
    for line in fulltext.splitlines():
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
    return fulltext, processedsections

def simplifytext(textsections, client, patientcontext=None):
    """
    Simplifies medical text sections with a focus on tasks, follow-ups, and importance, consolidating the output.
    """
    allactionitems = []
    for sectionname, text in textsections.items():
        prompt = (
            f"Patient Context: {patientcontext}\n\n"
            f"Medical Section Title: {sectionname}\n"
            f"Original Medical Text: {text}\n\n"
            f"Identify and list all the tasks the patient needs to do, any follow-up appointments they need to schedule or attend, and explain the importance of each item from the text below. "
            f"Use clear, plain language that a patient with limited medical knowledge can understand. "
            f"Format the output as a clear, bulleted list under the headings 'Tasks:', 'Follow-up Appointments:', and 'Importance:'. "
            f"Focus specifically on actionable steps for the patient's recovery and ongoing health management. "
            f"If information about importance is not explicitly stated, infer the importance based on the context (e.g., taking medication as prescribed is important for managing the condition).\n\n"
            f"Patient Action Items from {sectionname}:"
        )
        try:
            response = client.chat.completions.create(
                model="google/gemini-2.0-flash-thinking-exp:free",
                messages=[{"role": "user", "content": prompt}],
            )
            allactionitems.append(f"**{sectionname}:**\n{response.choices[0].message.content}\n\n")
        except Exception as e:
            allactionitems.append(f"[OpenRouter Error in {sectionname}] {e}\n\n")
    return "\n".join(allactionitems)

def extractkeyinfo(simplifiedtext):
    """
    Extracts key information from the consolidated simplified text.
    """
    keyinfo = []
    keywords = ['follow', 'call', 'take', 'return', 'appointment', 'contact', 'schedule', 'medication', 'important', 'need to']
    sentences = nltk.sent_tokenize(simplifiedtext)
    for sent in sentences:
        if any(keyword in sent.lower() for keyword in keywords):
            keyinfo.append(sent)
    return keyinfo

def evaluatereadability(simplifiedtext): 
    """
    Evaluates the readability of the consolidated simplified text.
    """
    score = textstat.flesch_reading_ease(simplifiedtext)
    return score

# Streamlit application
st.title("Discharge Instruction - Action Items")
uploadfile = st.file_uploader("Upload Discharge Instructions", type=["txt", "pdf"])

if uploadfile is not None:
    fulloriginaltext, data = loadandpreprocess(uploadfile)
    if data:
        st.subheader("Original Text")
        st.write(fulloriginaltext)

        st.subheader("Original Text Sections")
        for sectionname, text in data.items():
            st.write(f"**{sectionname}:** {text}")

        with st.spinner("Initializing OpenRouter client..."):
            client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=st.secrets["OPENROUTER_API_KEY"])

        patientcontext = st.text_input("Enter patient context (optional):")

        with st.spinner("Identifying Action Items..."):
            consolidatedactions = simplifytext(data, client, patientcontext=patientcontext)

        st.subheader("Key Action Items, Follow-ups, and Importance")
        st.write(consolidatedactions)

        # Optional: You can choose to display the extracted key info and readability if needed
        # keyinfo = extractkeyinfo(consolidated_actions)
        # st.subheader("Extracted Key Information")
        # st.write(keyinfo)
        #
        readability = evaluatereadability(consolidatedactions)
        st.subheader("Readability Score (Flesch Reading Ease)")
        st.write(readability)

    else:
        st.warning("No valid data found in the file.")
else:
    st.info("Please upload a discharge instructions file.")
