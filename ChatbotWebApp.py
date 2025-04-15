import os
import re
import json
import nltk
import PyPDF2
import textstat
import streamlit as st
from openai import OpenAI

# Ensure required NLTK packages are available
for package in ['punkt', 'punkt_tab']:
    try:
        nltk.data.find(package)
    except Exception:
        nltk.download(package)

llmcache = {}

def load_and_preprocess(uploadfile):
    """
    Reads the uploaded PDF or TXT file and returns a list of cleaned lines.
    """
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

    # Remove non-ASCII characters and extra whitespace from each line
    lines = [
        re.sub(r'\s+', ' ', re.sub(r'[^\x00-\x7F]+', ' ', line.strip()))
        for line in text.splitlines() if line.strip()
    ]
    return lines

def extract_and_simplify(text, client, patient_context=None):
    """
    Calls the LLM to process the discharge instructions.
    The prompt instructs the LLM to output JSON with these keys:
      - instructions, follow_ups, recommendations, summary, additional_attributes.
    """
    prompt = f"""
Patient Context (if any): {patient_context}

Discharge Instructions (Original):
{text}

Your Tasks:
1. Convert the above discharge instructions into plain, patient-friendly language.
2. Separate the content into these sections: instructions, follow-ups, recommendations.
3. For each item, label it with a priority (e.g., high, medium, low).
4. Provide a short summary of the entire discharge plan.
5. Include any additional helpful information if relevant.

Output must be valid JSON with this structure:
{{
  "instructions": [{{"text": "...", "priority": "..."}}, ...],
  "follow_ups": [{{"text": "...", "priority": "..."}}, ...],
  "recommendations": [{{"text": "...", "priority": "..."}}, ...],
  "summary": "...",
  "additional_attributes": "..."
}}
Use a Flesch Reading Ease target between 80 and 90.
Do not include any commentary outside of the JSON.
"""

    if prompt in llmcache:
        return llmcache[prompt]
    try:
        response = client.chat.completions.create(
            model="openrouter/auto",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            top_p=1
        )
        result = response.choices[0].message.content.strip()
        llmcache[prompt] = result
        return result
    except Exception as e:
        return f'[OpenRouter Error] {e}'

def evaluate_readability(text_block):
    """
    Returns the Flesch Reading Ease score for a given text.
    """
    return textstat.flesch_reading_ease(text_block)

def validate_output(original, simplified_json):
    """
    Performs a basic validation by comparing word counts of the original text
    and the combined simplified text.
    """
    original_len = len(original.split())
    try:
        data = json.loads(simplified_json)
        all_items = data.get("instructions", []) + data.get("follow_ups", []) + data.get("recommendations", [])
        simplified_texts = " ".join(item.get("text", "") for item in all_items)
        simplified_len = len(simplified_texts.split())
        ratio = simplified_len / max(original_len, 1)
        return f"Original word count: {original_len}, Simplified word count: {simplified_len}, Ratio: {ratio:.2f}"
    except json.JSONDecodeError:
        return "Could not parse JSON output from LLM."

################################
# Streamlit App Interface
################################

st.title("Discharge Instructions Simplifier")

# 1) File Uploader
uploaded_file = st.file_uploader("Upload Discharge Instructions", type=["txt", "pdf"])
if not uploaded_file:
    st.info("Please upload a discharge instructions file.")
    st.stop()

# 2) Load & display original instructions
lines = load_and_preprocess(uploaded_file)
if not lines:
    st.warning("No valid text found in the file.")
    st.stop()

original_text = " ".join(lines)

st.markdown("## Original Discharge Instructions")
for line in lines:
    st.write(line)
    st.write("")

# 3) Optional patient context
patient_context_input = st.text_input("Enter patient context (optional):")

# Use session_state to store the final context the user chooses
if "patient_context" not in st.session_state:
    st.session_state["patient_context"] = None

# 4) Two separate buttons for user actions
col1, col2 = st.columns(2)

with col1:
    if st.button("Apply Context"):
        # Store the user-provided context in session state
        st.session_state["patient_context"] = patient_context_input
        st.success("Patient context applied successfully.")

with col2:
    if st.button("Simplify Original Text"):
        with st.spinner("Initializing OpenRouter client..."):
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=st.secrets["OPENROUTER_API_KEY"]
            )

        with st.spinner("Extracting and Simplifying Instructions..."):
            # Retrieve the context from session state (could be None)
            final_context = st.session_state["patient_context"]
            response_json = extract_and_simplify(original_text, client, final_context)

        # Display the LLM response
        try:
            data = json.loads(response_json)

            st.markdown("### Simplified Instructions")
            instructions = data.get("instructions", [])
            if instructions:
                st.markdown("**Instructions:**")
                for item in instructions:
                    st.write(f"- {item.get('text', '')} (Priority: {item.get('priority', 'N/A')})")

            follow_ups = data.get("follow_ups", [])
            if follow_ups:
                st.markdown("**Follow-ups:**")
                for item in follow_ups:
                    st.write(f"- {item.get('text', '')} (Priority: {item.get('priority', 'N/A')})")

            recommendations = data.get("recommendations", [])
            if recommendations:
                st.markdown("**Recommendations:**")
                for item in recommendations:
                    st.write(f"- {item.get('text', '')} (Priority: {item.get('priority', 'N/A')})")

            summary = data.get("summary", "")
            if summary:
                st.markdown(f"**Summary:** {summary}")

            additional = data.get("additional_attributes", "")
            if additional:
                st.markdown(f"**Additional Attributes:** {additional}")

            # Evaluate readability & validation
            combined_text = (
                " ".join([i["text"] for i in instructions]) + " " +
                " ".join([f["text"] for f in follow_ups]) + " " +
                " ".join([r["text"] for r in recommendations])
            )
            readability = evaluate_readability(combined_text)
            st.subheader("Readability Score (Flesch Reading Ease)")
            st.write(readability)

            st.subheader("Validation Check")
            st.write(validate_output(original_text, response_json))

        except json.JSONDecodeError:
            st.error("Error parsing JSON output from the LLM. Please try again or check the input.")
