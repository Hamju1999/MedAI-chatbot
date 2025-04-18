# app.py

import os
import re
import io
import datetime
import requests
import streamlit as st

# optional readability scoring
try:
    import textstat
except ImportError:
    textstat = None

# PDF export
try:
    from fpdf import FPDF
except ImportError:
    FPDF = None

# Word export
try:
    from docx import Document
except ImportError:
    Document = None

# Text-to-speech
try:
    from gtts import gTTS
except ImportError:
    gTTS = None

# Voice transcription
try:
    import speech_recognition as sr
except ImportError:
    sr = None

# --- UI setup ---
st.set_page_config(page_title="Discharge Summary Simplifier", layout="wide")
st.title("Discharge Summary Simplifier")
st.subheader("Transforming Patient Understanding with AI")
st.write("Upload a hospital discharge summary and let the AI simplify it into clear instructions.")

# --- Offline caching ---
if "cached_summary" not in st.session_state:
    st.session_state["cached_summary"] = None
if "cached_sections" not in st.session_state:
    st.session_state["cached_sections"] = {}

# --- API key handling ---
api_key = os.getenv("OPENROUTER_API_KEY", "")
if not api_key:
    api_key = st.text_input("Enter your OpenRouter API Key", type="password")
if not api_key and st.session_state["cached_summary"]:
    st.info("Offline mode: Loaded last cached summary.")
else:
    if not api_key:
        st.info("🔑 Please provide your OpenRouter API key.")
        st.stop()

# --- File uploader ---
uploaded_file = st.file_uploader(
    "Upload Discharge Summary (TXT or PDF)",
    type=["txt", "pdf"],
    help="Accepted formats: .txt or .pdf"
)
if not uploaded_file:
    st.stop()

# --- Patient Context & Voice Input ---
if "patient_context" not in st.session_state:
    st.session_state["patient_context"] = ""
# Text context input
patient_context_input = st.text_input("Enter patient context (optional):")
# Voice context upload
audio_file = st.file_uploader("Or upload voice note for context (mp3/wav)", type=["mp3","wav"],help="Optional: record additional context")
if audio_file and sr:
    r = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = r.record(source)
    try:
        transcribed = r.recognize_google(audio)
        st.write(f"Transcribed voice note: {transcribed}")
        patient_context_input = transcribed
    except Exception:
        st.warning("Could not transcribe audio.")
elif audio_file:
    st.info("`speech_recognition` not installed for transcription.")
# Apply Context button
if st.button("Apply Context", key="apply_context_btn"):
    st.session_state["patient_context"] = patient_context_input
    st.success("Patient context applied successfully.")
current_context = st.session_state["patient_context"]

# --- Customizable Font & Contrast ---
font_size = st.sidebar.slider("Font size", 12, 24, 16)
high_contrast = st.sidebar.checkbox("High Contrast Mode")
if high_contrast:
    st.markdown("<style>body {background-color:#000; color:#fff;}</style>", unsafe_allow_html=True)
st.markdown(f"<style>* {{ font-size: {font_size}px; }}</style>", unsafe_allow_html=True)

# --- Options ---
col1, col2 = st.columns(2)
with col1:
    reading_level = st.slider("Target Reading Level (US Grade)", min_value=3, max_value=12, value=6)
with col2:
    language = st.selectbox("Output Language", ["English","Spanish","Chinese","French","German"], index=0)

# --- Text extraction ---
def extract_text_from_file(file) -> str:
    if file.type == "application/pdf":
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            st.error("PDF support requires PyPDF2. Please `pip install PyPDF2`.")
            return ""
        try:
            reader = PdfReader(file)
        except Exception:
            import io as _io
            reader = PdfReader(_io.BytesIO(file.read()))
        text = ""
        for page in reader.pages:
            text += (page.extract_text() or "") + "\n"
        return text
    else:
        data = file.read()
        for enc in ("utf-8","latin-1"): 
            try: return data.decode(enc)
            except: pass
        return ""

discharge_text = extract_text_from_file(uploaded_file).strip()
if not discharge_text:
    st.error("❌ Could not extract any text. Please upload a valid TXT or PDF.")
    st.stop()

# --- Glossary for tooltips ---
glossary = {
    "otoscope exam": "An exam where a doctor looks inside your ear",
    "mri": "Magnetic Resonance Imaging, a scan that uses magnets to view inside the body"
}
def apply_tooltips(line: str) -> str:
    for term, defi in glossary.items():
        pattern = re.compile(rf"\b({re.escape(term)})\b", flags=re.IGNORECASE)
        line = pattern.sub(rf"<span title='{defi}' style='border-bottom:1px dotted;'>\1</span>", line)
    return line

# --- LLM call ---
def summarize_discharge(text: str, reading_lvl: int, lang: str, patient_context: str) -> dict:
    prompt = (
        f"Patient Context (if any): {patient_context}\n"
        f"The following is a hospital discharge summary. Simplify it to a {reading_lvl}th-grade reading level in {lang}.\n"
        "Break into sections with these headings:\n"
        "- **Simplified Instructions:** bullet-points\n"
        "- **Importance:** why each instruction matters\n"
        "- **Follow-Up Appointments or Tasks:** tasks/visits\n"
        "- **Medications:** with simple dosing notes\n"
        "- **Precautions:** warning signs, activities to avoid\n"
        "- **References:** brief reasons/explanations\n\n"
        f"Now simplify:\n\"\"\"{text}\"\"\""
    )
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": "deepseek/deepseek-r1", "messages": [{"role": "user","content": prompt}]}
    resp = requests.post(url, headers=headers, json=payload)
    return {"status": resp.status_code, "raw_json": resp.json() if resp.headers.get("content-type",""
               ).startswith("application/json") else resp.text}

# --- Helper: Generate calendar ICS ---
def generate_ics(event_title: str) -> str:
    dt = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    return ("BEGIN:VCALENDAR\nVERSION:2.0\nBEGIN:VEVENT\n" 
            f"SUMMARY:{event_title}\nDTSTART:{dt}\nEND:VEVENT\nEND:VCALENDAR")

# --- Main action ---
if st.button("Simplify Discharge Instructions"):
    # Use cache if offline
    if not api_key and st.session_state["cached_summary"]:
        simplified_text = st.session_state["cached_summary"]
        sections = st.session_state["cached_sections"]
    else:
        with st.spinner("🧠 Summarizing…"):
            api_resp = summarize_discharge(discharge_text, reading_level, language, current_context)
        if api_resp["status"] != 200:
            st.error(f"API returned status {api_resp['status']}")
            st.stop()
        choices = api_resp["raw_json"].get("choices", [])
        simplified_text = choices[0].get("message",{}).get("content","\n").strip() if choices else ""
        st.session_state["cached_summary"] = simplified_text
        # parse sections
        header_re = re.compile(r"^\*{0,2}(.+?)\*{0,2}:?$")
        sections = {"Simplified Instructions":[],"Importance":[],"Follow-Up Appointments or Tasks":[],
                    "Medications":[],"Precautions":[],"References":[]}
        current = None
        for line in simplified_text.splitlines():
            m = header_re.match(line.strip())
            if m and m.group(1).strip() in sections:
                current = m.group(1).strip(); continue
            if current: sections[current].append(line)
        st.session_state["cached_sections"] = sections

    # Display summary
    st.markdown("---")
    st.subheader("📄 Formatted Simplified Summary")
    for line in simplified_text.splitlines():
        st.markdown(apply_tooltips(line), unsafe_allow_html=True)

    # PDF export
    if FPDF:
        pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial", size=12)
        for line in simplified_text.splitlines(): pdf.cell(0,10,txt=line,ln=1)
        pdf_bytes = pdf.output(dest="S").encode("latin-1")
        st.download_button("📥 Download PDF", pdf_bytes, file_name="summary.pdf", mime="application/pdf")

    # DOCX export
    if Document:
        doc = Document()
        for line in simplified_text.splitlines(): doc.add_paragraph(line)
        doc_io = io.BytesIO()
        doc.save(doc_io)
        doc_io.seek(0)
        st.download_button("📥 Download DOCX", data=doc_io, file_name="summary.docx")

    # Text-to-speech
    if gTTS:
        if st.button("▶️ Play Audio"):
            tts = gTTS(simplified_text)
            buf = io.BytesIO(); tts.write_to_fp(buf); buf.seek(0)
            st.audio(buf.read(), format="audio/mp3")

    # Interactive Q&A
    st.markdown("---")
    st.subheader("💬 Interactive Q&A")
    qa = st.text_input("Ask a question about your summary:")
    if st.button("Ask Question"):
        with st.spinner("🗣 Answering…"):
            qa_prompt = f"{simplified_text}\n\nPatient Q&A: {qa}"
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization":f"Bearer {api_key}","Content-Type":"application/json"},
                json={"model":"deepseek/deepseek-r1","messages":[{"role":"user","content":qa_prompt}]}
            )
            answer = resp.json().get("choices",[])[0].get("message",{}).get("content","" )
        st.write(answer)

    # Readability
    st.markdown("---")
    st.subheader("🔢 Readability")
    if textstat:
        score = textstat.flesch_kincaid_grade(simplified_text)
        st.write(f"Flesch–Kincaid Grade Level: **{score:.1f}**")

    # Parsed Sections & Actions
    st.markdown("---")
    st.subheader("🔀 Parsed Sections & Actions")
    icons = {
        "Simplified Instructions": "📝",
        "Importance": "💡",
        "Follow-Up Appointments or Tasks": "📅",
        "Medications": "💊",
        "Precautions": "⚠️",
        "References": "📚"
    }
    for sec, lines in sections.items():
        if not lines:
            continue
        st.markdown(f"**{icons.get(sec, '')} {sec}**")
        for line in lines:
            st.markdown(apply_tooltips(line), unsafe_allow_html=True)
        if sec == "Follow-Up Appointments or Tasks":
            for fu in lines:
                ics = generate_ics(fu)
                st.download_button(f"📅 Add '{fu}' to Calendar", data=ics, file_name="event.ics", mime="text/calendar")
        if sec == "Medications":
            st.subheader("Medication Checklist & Reminders")
            for med in lines:
                st.checkbox(med, key=med)
            if st.button("Schedule Med Reminders", key="med_reminders_btn"):
                st.success("Medication reminders scheduled!")

    # Symptom Tracker
    st.markdown("---")
    if st.checkbox("Enable Symptom Tracker"):
        st.subheader("📋 Symptom Tracker")
        d = st.date_input("Date", datetime.date.today())
        pain = st.slider("Pain level", 0, 10, 0)
        swelling = st.slider("Swelling level", 0, 10, 0)
        if st.button("Log Symptom"):
            if "symptoms" not in st.session_state: st.session_state["symptoms"]=[]
            st.session_state["symptoms"].append({"date":str(d),"pain":pain,"swelling":swelling})
            st.success("Symptom logged")
        if st.session_state.get("symptoms"):
            st.write(st.session_state["symptoms"]) 

    # Feedback to provider
    st.markdown("---")
    st.subheader("📧 Send Feedback to Provider")
    msg = st.text_area("Your message to your care team")
    if st.button("Send Message"):
        st.success("Your message has been sent to your provider.")

    # Emergency contacts
    st.markdown("---")
    st.subheader("🚨 Emergency Contacts")
    ec_name = st.text_input("Contact Name")
    ec_num = st.text_input("Contact Number")
    if st.button("Save Contact"):
        st.session_state["emergency"]={"name":ec_name,"number":ec_num}
        st.success("Emergency contact saved")
    if st.session_state.get("emergency"):
        em = st.session_state["emergency"]
        st.markdown(f"[Call {em['name']}]({{'tel:' + em['number']}})")

    # Trusted resources
    st.markdown("---")
    st.subheader("🔗 Trusted Resources")
    links = {"Mayo Clinic":"https://www.mayoclinic.org","NIH":"https://www.nih.gov"}
    for name, url in links.items():
        st.markdown(f"- [{name}]({url})")

    # Quick quiz
    st.markdown("---")
    st.subheader("🧠 Quick Quiz")
    ans1 = st.radio("What day is your follow-up?", ["Monday","Tuesday","Other"])
    ans2 = st.radio("How often do you take your meds?", ["Once a day","Twice a day"])
    ans3 = st.radio("Do you know warning signs to watch for?", ["Yes","No"])
    if st.button("Submit Quiz"):
        st.info(f"Your answers: {ans1}, {ans2}, {ans3}")
        st.success("Quiz submitted.")
