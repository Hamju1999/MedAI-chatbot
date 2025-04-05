import os
import PyPDF2
import nltk
import re
import textstat
import nltk
import streamlit as st
from openai import OpenAI

# Ensure necessary NLTK data is downloaded
nltkpackages = ['punkt', 'punkt_tab']
for package in nltkpackages:
    try:
        nltk.data.find(package)
    except Exception as e:
        print(f"Error finding {package} data: {e}")
        nltk.download(package)

def loadandpreprocess(uploadfile):
    """
    Loads text data, performs preprocessing.
    Returns the full original text.
    """
    _, ext = os.path.splitext(uploadfile.name)
    fulltext = ""  
    if ext.lower() == ".pdf":
        try:
            reader = PyPDF2.PdfReader(uploadfile)
            textpages = [page.extract_text() or "" for page in reader.pages]
            fulltext = "".join(textpages)
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
    else:
        try:
            fulltext = uploadfile.read().decode("utf-8")
        except Exception as e:
            st.error(f"Error decoding text file: {e}")
    processedtext = re.sub(r'\s+', ' ', re.sub(r'[^\x00-\x7F]+', ' ', fulltext.strip()))
    return processedtext

def simplifytext(fulltext, client, patientcontext=None):
    """
    Simplifies the entire medical text with a focus on tasks, follow-ups, and importance, consolidating the output.
    """
    prompt = (
        f"Patient Context: {patientcontext}\n\n"
        f"Original Medical Text:\n{fulltext}\n\n"
        f"From the discharge instructions above, identify and list all the tasks the patient needs to do, any follow-up appointments they need to schedule or attend, and explain the importance of each item. "
        f"Use clear, plain language that a patient with limited medical knowledge can understand. "
        f"Organize the output into three main sections: 'Tasks:', 'Follow-up Appointments:', and 'Importance:'. "
        f"Under each section, use a bulleted list for each item. Focus specifically on actionable steps for the patient's recovery and ongoing health management. "
        f"If the importance of a task or follow-up is not explicitly stated, infer it based on the medical context.\n\n"
        f"Patient Action Items:"
    )
    try:
        response = client.chat.completions.create(
            model="google/gemini-2.0-flash-thinking-exp:free",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[OpenRouter Error] {e}"

def extractkeyinfo(simplifiedtext):
    """
    Extracts key information from the simplified text.
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
    Evaluates the readability of the simplified text.
    """
    score = textstat.flesch_reading_ease(simplifiedtext)
    return score

# Streamlit application
st.title("Discharge Instruction - Action Items")
uploadfile = st.file_uploader("Upload Discharge Instructions", type=["txt", "pdf"])

if uploadfile is not None:
    fulloriginaltext = loadandpreprocess(uploadfile)
    if fulloriginaltext:
        st.subheader("Original Text")
        st.write(fulloriginaltext)

        with st.spinner("Initializing OpenRouter client..."):
            client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=st.secrets["OPENROUTER_API_KEY"])

        patientcontext = st.text_input("Enter patient context (optional):")

        with st.spinner("Identifying Action Items..."):
            consolidatedactions = simplifytext(fulloriginaltext, client, patientcontext=patientcontext)

        st.subheader("Key Action Items, Follow-ups, and Importance")
        st.write(consolidatedactions)

        # Optional: You can choose to display the extracted key info and readability if needed
        # keyinfo = extractkeyinfo(consolidatedactions)
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
