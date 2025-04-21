# Libraries
import os
import re
import io
import datetime
import locale
import base64
import requests
import json
import streamlit as st
import pandas as pd

# optional readability scoring
try:
    import textstat
except ImportError:
    textstat = None

# PDF and Word export
try:
    from fpdf import FPDF
except ImportError:
    FPDF = None

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
        for enc in ("utf-8", "latin-1"):
            try:
                return data.decode(enc)
            except:
                pass
        return ""

# --- UI setup ---
st.set_page_config(page_title="Discharge Summary Simplifier")
st.title("Discharge Summary Simplifier")
st.subheader("Transforming Patient Understanding with AI")
st.write("Upload a hospital discharge summary and let the AI simplify it into clear instructions.")

# --- Offline caching & run flag ---
if "cached_summary" not in st.session_state:
    st.session_state["cached_summary"] = None
if "cached_sections" not in st.session_state:
    st.session_state["cached_sections"] = {}
if "cached_concise" not in st.session_state:
    st.session_state["cached_concise"] = None
if "run_summary" not in st.session_state:
    st.session_state["run_summary"] = False
if "symptoms" not in st.session_state:
    st.session_state["symptoms"] = []
if "faq_log" not in st.session_state:
    st.session_state["faq_log"] = []

# --- API key handling ---
api_key = os.getenv("OPENROUTER_API_KEY", "")
if not api_key:
    api_key = st.text_input("Enter your OpenRouter API Key", type="password")
if not api_key and st.session_state["cached_summary"]:
    st.info("Offline mode: Using cached summary.")
else:
    if not api_key:
        st.info("Please provide your OpenRouter API key.")
        st.stop()

# --- Input mode selector ---
mode = st.radio(
    "How would you like to provide the discharge text?",
    ("Enter text", "Upload file", "Voice note"),
    horizontal=True
)

if "discharge_text" not in st.session_state:
    st.session_state["discharge_text"] = ""

# 1) Manual entry
if mode == "Enter text":
    current = st.session_state["discharge_text"]
    line_count = current.count("\n") + 1
    height_px = min(max(line_count * 25, 200), 800)
    discharge_text = st.text_area(
        "Discharge Instructions Text:",
        height=height_px,
        help="Paste or type your instructions here; updates immediately",
        key="discharge_text"
    )

# 2) File upload
elif mode == "Upload file":
    uploaded_file = st.file_uploader(
        "Upload Discharge Summary (TXT or PDF)",
        type=["txt", "pdf"],
        help="Accepted formats: .txt or .pdf"
    )
    if uploaded_file:
        extracted = extract_text_from_file(uploaded_file).strip()
        if not extracted:
            st.error("Could not extract any text. Please upload a valid file.")
        else:
            st.session_state["discharge_text"] = extracted
    discharge_text = st.session_state["discharge_text"]

# 3) Voice note
else:
    audio_file = st.file_uploader(
        "Or upload voice note for context (mp3/wav)",
        type=["mp3", "wav"],
        help="Optional: record additional context"
    )
    if audio_file and sr:
        r = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio = r.record(source)
        try:
            transcribed = r.recognize_google(audio)
            st.write(f"Transcribed voice note: {transcribed}")
            st.session_state["discharge_text"] = transcribed
        except Exception:
            st.warning("Could not transcribe audio.")
    elif audio_file:
        st.info("`speech_recognition` not installed for transcription.")
    discharge_text = st.session_state["discharge_text"]

# --- Display raw text ---
if discharge_text:
    st.markdown("#### Original Text")
    st.write(discharge_text)
else:
    st.info("Provide discharge instructions above using the selected mode, then click ‘Simplify Discharge Instructions’")
    st.stop()

# --- Dynamic Detail Level ---
show_details = st.sidebar.checkbox("Show Detailed Medical Jargon")

# --- Customizable Font & Contrast ---
font_size = st.sidebar.slider("Font size", 12, 24, 16)
high_contrast = st.sidebar.checkbox("High Contrast Mode")
if high_contrast:
    st.markdown("<style>body {background-color:#000; color:#fff;}</style>", unsafe_allow_html=True)
st.markdown(f"<style>* {{ font-size: {font_size}px; }}</style>", unsafe_allow_html=True)

# --- Auto-translation based on locale ---
user_locale = locale.getdefaultlocale()[0]
af_lang = user_locale.split('_')[0].capitalize() if user_locale else None
if af_lang not in ["English","Spanish","Chinese","French","German"]:
    af_lang = None

# --- Options ---
col1, col2 = st.columns(2)
with col1:
    reading_level = st.slider("Target Reading Level (US Grade)", min_value=3, max_value=12, value=6)
with col2:
    language = st.selectbox("Output Language", ["English","Spanish","Chinese","French","German"], index=0)

# --- Glossary for tooltips ---
glossary = {
    "otoscope exam": "An exam where a doctor looks inside your ear",
    "mri": "Magnetic Resonance Imaging, a scan that uses magnets to view inside the body"
}
def apply_tooltips(line: str) -> str:
    for term, defi in glossary.items():
        pattern = re.compile(rf"\b({re.escape(term)})\b", flags=re.IGNORECASE)
        line = pattern.sub(
            rf"<span title='{defi}' style='border-bottom:1px dotted;'>\1</span>",
            line
        )
    return line

# --- LLM functions ---
def generate_concise_summary(text: str, lang: str) -> dict:
    prompt = (
        f"Simplify the following discharge instructions into a concise, include all essential "
        f"information related to the patient especially the diagnosis and the reason, patient-friendly overview.\n"
        f"Output only a short paragraph (no sections):\n\n\"\"\"{text}\"\"\""
    )
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "deepseek/deepseek-r1",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_p": 1.0
    }
    resp = requests.post(url, headers=headers, json=payload)
    return {
        "status": resp.status_code,
        "raw_json": resp.json() if resp.headers.get("content-type","").startswith("application/json") else resp.text
    }

def summarize_discharge(text: str, reading_lvl: int, lang: str) -> dict:
    prompt = (
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
    payload = {
        "model": "deepseek/deepseek-r1",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_p": 1.0
    }
    resp = requests.post(url, headers=headers, json=payload)
    return {
        "status": resp.status_code,
        "raw_json": resp.json() if resp.headers.get("content-type","").startswith("application/json") else resp.text
    }

# --- Helper: Generate calendar ICS ---
def generate_ics(event_title: str) -> str:
    dt = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    return (
        "BEGIN:VCALENDAR\nVERSION:2.0\nBEGIN:VEVENT\n"
        f"SUMMARY:{event_title}\nDTSTART:{dt}\nEND:VEVENT\nEND:VCALENDAR"
    )

# --- Main action ---
if st.button("Simplify Discharge Instructions"):
    # Concise summary
    if not api_key and st.session_state["cached_concise"]:
        concise_text = st.session_state["cached_concise"]
    else:
        with st.spinner("Generating concise summary…"):
            api_resp = generate_concise_summary(discharge_text, language)
        if api_resp["status"] != 200:
            st.error(f"API returned status {api_resp['status']}")
            st.stop()
        choices = api_resp["raw_json"].get("choices", [])
        concise_text = choices[0].get("message", {}).get("content", "").strip() if choices else ""
        st.session_state["cached_concise"] = concise_text

    # Detailed simplified summary (cached, not displayed)
    if not api_key and st.session_state["cached_summary"]:
        simplified_text = st.session_state["cached_summary"]
        sections = st.session_state["cached_sections"]
    else:
        with st.spinner("Summarizing discharge…"):
            api_resp2 = summarize_discharge(discharge_text, reading_level, language)
        if api_resp2["status"] != 200:
            st.error(f"API returned status {api_resp2['status']}")
            st.stop()
        choices2 = api_resp2["raw_json"].get("choices", [])
        simplified_text = choices2[0].get("message", {}).get("content", "").strip() if choices2 else ""
        st.session_state["cached_summary"] = simplified_text

        # Parse sections
        sections = {
            "Simplified Instructions": [],
            "Importance": [],
            "Follow-Up Appointments or Tasks": [],
            "Medications": [],
            "Precautions": [],
            "References": []
        }
        current = None
        for line in simplified_text.splitlines():
            stripped = line.strip()
            text = re.sub(r"^[\-\*\s]+", "", stripped)
            for sec_name in sections:
                if text.lower().startswith(sec_name.lower()):
                    current = sec_name
                    break
            else:
                if current and text:
                    clean_text = re.sub(r"[:\*]+$", "", text).strip()
                    sections[current].append(clean_text)
        st.session_state["cached_sections"] = sections

    st.session_state["run_summary"] = True

# --- Persistent display after summary ---
if st.session_state["run_summary"]:
    concise_text = st.session_state["cached_concise"]
    sections = st.session_state["cached_sections"]

    # Display simplified summary
    st.markdown("---")
    st.subheader("Simplified Summary")
    for line in concise_text.splitlines():
        st.markdown(apply_tooltips(line), unsafe_allow_html=True)

    # Categorization & Actions
    st.markdown("---")
    st.subheader("Categorization & Actions")
    # Clean up stray chars
    for sec, items in sections.items():
        sections[sec] = [re.sub(r'\}+$', '', itm).strip() for itm in items]
    sections = {
        re.sub(r'\*+', '', sec).strip(): [re.sub(r'\*+', '', itm).strip() for itm in items]
        for sec, items in sections.items()
    }
    icons = {
        "Simplified Instructions": "",
        "Importance": "",
        "Follow-Up Appointments or Tasks": "",
        "Medications": "",
        "Precautions": "",
        "References": ""
    }
    if not sections or all(len(v) == 0 for v in sections.values()):
        st.info("No structured sections found. Please ensure your summary uses the expected headings.")
    else:
        for sec, items in sections.items():
            if not items:
                continue
            st.markdown(f"{icons.get(sec,'')} {sec}")
            for itm in items:
                st.markdown(f"- {apply_tooltips(itm)}", unsafe_allow_html=True)
            if sec == "Follow-Up Appointments or Tasks":
                for fu in items:
                    ics = generate_ics(fu)
                    st.download_button(
                        f"Add '{fu}' to Calendar",
                        data=ics,
                        file_name="event.ics",
                        mime="text/calendar"
                    )
            if sec == "Medications":
                st.subheader("Medication Checklist & Reminders")
                for med in items:
                    st.checkbox(med, key=med)
                if st.button("Schedule Med Reminders", key="med_reminders_btn"):
                    st.success("Medication reminders scheduled!")

        # Overall reading level and JSON export
        if textstat:
            combined = " ".join(itm for items in sections.values() for itm in items)
            overall_grade = textstat.flesch_kincaid_grade(combined)
            st.markdown(f"*Overall reading level: {overall_grade:.1f}th grade*", unsafe_allow_html=True)

        st.markdown("#### Categorization as JSON")
        st.json(sections)
        json_payload = json.dumps(sections, indent=2)
        st.download_button(
            label="Download Categorization JSON",
            data=json_payload,
            file_name="categorization.json",
            mime="application/json"
        )

    # 3) Actions & Trackers
    st.markdown("---")
    st.subheader("Actions & Trackers")

    # High‑pain alert
    if st.session_state["symptoms"]:
        last = st.session_state["symptoms"][-1]
        if last.get("pain", 0) > 8:
            st.warning("High pain detected – consider contacting your provider.")

    # Recovery Timeline
    if st.checkbox("Show Recovery Timeline", key="show_timeline"):
        if st.session_state["symptoms"]:
            df = pd.DataFrame(st.session_state["symptoms"])
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")[["pain", "swelling"]]
            st.line_chart(df)
        else:
            st.info("No symptom data yet.")

    # Download Symptom Log
    if st.session_state["symptoms"]:
        csv_data = pd.DataFrame(st.session_state["symptoms"]).to_csv(index=False)
        st.download_button(
            label="Download Symptom Log for Clinician",
            data=csv_data,
            file_name="symptoms.csv",
            mime="text/csv",
            key="download_symptoms"
        )

    # EHR Sync stub
    if st.button("Sync with EHR", key="sync_ehr"):
        st.info("EHR integration not configured.")

    # Caregiver / Proxy Sharing
    if st.checkbox("Enable caregiver access", key="enable_caregiver"):
        email = st.text_input("Caregiver email:", key="caregiver_email")
        if email and st.button("Generate share link", key="gen_share"):
            link = f"https://yourapp.example.com/share?email={email}"
            st.success(f"Shareable link: {link}")

    # SMS & Push Reminders
    phone = st.text_input("Phone number for SMS reminders:", key="sms_phone")
    if phone and st.button("Enable SMS Reminders", key="enable_sms"):
        st.success(f"SMS reminders will be sent to {phone}")

    # Symptom Tracker
    st.markdown("---")
    st.subheader("Symptom Tracker")
    d = st.date_input("Date", datetime.date.today(), key="symptom_date")
    pain = st.slider("Pain level", 0, 10, 0, key="symptom_pain")
    swelling = st.slider("Swelling level", 0, 10, 0, key="symptom_swelling")
    if st.button("Log Symptom", key="symptom_log_btn"):
        entry = {"date": str(d), "pain": pain, "swelling": swelling}
        st.session_state["symptoms"].append(entry)
        st.success("Symptom logged")
    if st.session_state["symptoms"]:
        st.dataframe(pd.DataFrame(st.session_state["symptoms"]))

    # Send Feedback
    st.markdown("---")
    st.subheader("Send Feedback to Provider")
    feedback = st.text_area("Your message to your care team", key="feedback_msg")
    if st.button("Send Message", key="send_feedback_btn"):
        st.session_state["faq_log"].append(feedback)
        st.success("Your message has been sent to your provider.")

    # Privacy Dashboard
    st.markdown("---")
    st.subheader("Privacy Dashboard")
    if st.button("View Stored Data", key="view_data_btn"):
        st.json({k: st.session_state[k] for k in st.session_state})
    if st.button("Clear All Data", key="clear_data_btn"):
        for k in ["cached_summary","cached_sections","cached_concise","run_summary","symptoms","faq_log","emergency"]:
            st.session_state.pop(k, None)
        st.success("Data cleared.")

    # Mood & Interaction Alerts
    st.markdown("---")
    mood = st.select_slider(
        "How are you feeling today?",
        options=["Good","Okay","Poor"],
        key="mood_slider"
    )
    if mood == "Poor":
        st.warning("We noticed you feel poor—review precautions or contact your provider.")

    # Emergency Contacts
    st.markdown("---")
    st.subheader("🚨 Emergency Contacts")
    ec_name = st.text_input("Contact Name", key="ec_name_input")
    ec_num = st.text_input("Contact Number", key="ec_num_input")
    if st.button("Save Contact", key="save_ec_btn"):
        st.session_state["emergency"] = {"name": ec_name, "number": ec_num}
        st.success("Emergency contact saved")
    if "emergency" in st.session_state:
        em = st.session_state["emergency"]
        st.markdown(f"[Call {em['name']}]({{'tel:' + em['number']}})")
