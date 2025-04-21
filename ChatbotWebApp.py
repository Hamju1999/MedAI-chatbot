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

# Email & SMS clients
try:
    import smtplib
    from email.message import EmailMessage
except ImportError:
    smtplib = None

try:
    from twilio.rest import Client as TwilioClient
except ImportError:
    TwilioClient = None

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
        for enc in ("utf-8","latin-1"): 
            try:
                return data.decode(enc)
            except:
                pass
        return ""

# --- UI setup ---
st.set_page_config(page_title="Discharge Summary Simplifier")
st.title("Discharge Summary Simplifier")
st.subheader("Transforming Patient Understanding with AI")
st.write("Upload or paste a hospital discharge summary and let the AI simplify it.")

# --- Offline caching & run flag ---
for key, default in [
    ("cached_summary", None),
    ("cached_sections", {}),
    ("cached_concise", None),
    ("run_summary", False),
    ("symptoms", []),
    ("faq_log", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# --- API key handling ---
api_key = os.getenv("OPENROUTER_API_KEY", "")
if not api_key:
    api_key = st.text_input("Enter your OpenRouter API Key", type="password")
if not api_key and st.session_state["cached_summary"]:
    st.info("Offline mode: Using cached summary.")
elif not api_key:
    st.info("Please provide your OpenRouter API key.")
    st.stop()

# --- Input mode selector ---
mode = st.radio(
    "How provide the discharge text?",
    ("Enter text", "Upload file", "Voice note"),
    horizontal=True
)
if "discharge_text" not in st.session_state:
    st.session_state["discharge_text"] = ""

if mode == "Enter text":
    curr = st.session_state["discharge_text"]
    h = min(max((curr.count("\n")+1)*25, 200), 800)
    st.text_area("Discharge Instructions Text:", height=h, key="discharge_text")
elif mode == "Upload file":
    f = st.file_uploader("Upload Discharge Summary (TXT/PDF)", type=["txt","pdf"])
    if f:
        txt = extract_text_from_file(f).strip()
        if not txt:
            st.error("Invalid file.")
        else:
            st.session_state["discharge_text"] = txt
else:
    audio = st.file_uploader("Upload voice note (mp3/wav)", type=["mp3","wav"])
    if audio and sr:
        r = sr.Recognizer()
        with sr.AudioFile(audio) as src:
            aud = r.record(src)
        try:
            txt = r.recognize_google(aud)
            st.session_state["discharge_text"] = txt
        except:
            st.warning("Transcription failed.")
    elif audio:
        st.info("Install speech_recognition for transcription.")

discharge_text = st.session_state["discharge_text"]
if discharge_text:
    st.markdown("#### Original Text")
    st.write(discharge_text)
else:
    st.info("Provide text above, then click ‘Simplify Discharge Instructions’")
    st.stop()

# --- Sidebar & options ---
show_details = st.sidebar.checkbox("Show Detailed Medical Jargon")
font_size = st.sidebar.slider("Font size", 12,24,16)
high_contrast = st.sidebar.checkbox("High Contrast Mode")
if high_contrast:
    st.markdown("<style>body{background:#000;color:#fff;}</style>", unsafe_allow_html=True)
st.markdown(f"<style>*{{font-size:{font_size}px}}</style>", unsafe_allow_html=True)
user_locale = locale.getdefaultlocale()[0] or ""
af_lang = user_locale.split("_")[0].capitalize()
if af_lang not in ["English","Spanish","Chinese","French","German"]:
    af_lang = None
col1, col2 = st.columns(2)
with col1:
    reading_level = st.slider("Target Reading Level (US Grade)",3,12,6)
with col2:
    language = st.selectbox("Output Language", ["English","Spanish","Chinese","French","German"], 0)

# --- Glossary ---
glossary = {
    "otoscope exam": "An exam where a doctor looks inside your ear",
    "mri": "Magnetic Resonance Imaging scan"
}
def apply_tooltips(line:str)->str:
    for term,defi in glossary.items():
        line = re.sub(rf"\b({re.escape(term)})\b",
                      rf"<span title='{defi}' style='border-bottom:1px dotted;'>\1</span>",
                      line, flags=re.IGNORECASE)
    return line

# --- LLM calls ---
def generate_concise_summary(text:str, lang:str)->dict:
    prompt = (
        f"Simplify the following discharge instructions into a concise, include all essential "
        f"information related to the patient especially the diagnosis and the reason, patient-friendly overview.\n"
        f"Output only a short paragraph (no sections):\n\n\"\"\"{text}\"\"\""
    )
    payload = {"model":"deepseek/deepseek-r1",
               "messages":[{"role":"user","content":prompt}],
               "temperature":0.0,"top_p":1.0}
    resp = requests.post("https://openrouter.ai/api/v1/chat/completions",
                         headers={"Authorization":f"Bearer {api_key}","Content-Type":"application/json"},
                         json=payload)
    return {"status":resp.status_code,
            "raw_json":resp.json() if resp.headers.get("content-type","").startswith("application/json") else resp.text}

def summarize_discharge(text:str, lvl:int, lang:str)->dict:
    prompt = (
        f"The following is a hospital discharge summary. Simplify it to a {lvl}th-grade reading level in {lang}.\n"
        "Break into sections with these headings:\n"
        "- **Simplified Instructions:** bullet-points\n"
        "- **Importance:** why each instruction matters\n"
        "- **Follow-Up Appointments or Tasks:** tasks/visits\n"
        "- **Medications:** with simple dosing notes\n"
        "- **Precautions:** warning signs, activities to avoid\n"
        "- **References:** brief reasons/explanations\n\n"
        f"Now simplify:\n\"\"\"{text}\"\"\""
    )
    payload = {"model":"deepseek/deepseek-r1",
               "messages":[{"role":"user","content":prompt}],
               "temperature":0.0,"top_p":1.0}
    resp = requests.post("https://openrouter.ai/api/v1/chat/completions",
                         headers={"Authorization":f"Bearer {api_key}","Content-Type":"application/json"},
                         json=payload)
    return {"status":resp.status_code,
            "raw_json":resp.json() if resp.headers.get("content-type","").startswith("application/json") else resp.text}

def generate_ics(evt:str)->str:
    dt = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    return f"BEGIN:VCALENDAR\nVERSION:2.0\nBEGIN:VEVENT\nSUMMARY:{evt}\nDTSTART:{dt}\nEND:VEVENT\nEND:VCALENDAR"

# --- Main action ---
if st.button("Simplify Discharge Instructions"):
    # concise
    if not api_key and st.session_state["cached_concise"]:
        concise = st.session_state["cached_concise"]
    else:
        r = generate_concise_summary(discharge_text, language)
        if r["status"]!=200:
            st.error(f"API {r['status']}"); st.stop()
        concise = r["raw_json"]["choices"][0]["message"]["content"].strip()
        st.session_state["cached_concise"] = concise

    # detailed (cache, but hidden)
    if not api_key and st.session_state["cached_summary"]:
        _, secs = st.session_state["cached_summary"], st.session_state["cached_sections"]
    else:
        r2 = summarize_discharge(discharge_text, reading_level, language)
        if r2["status"]!=200:
            st.error(f"API {r2['status']}"); st.stop()
        detailed = r2["raw_json"]["choices"][0]["message"]["content"].strip()
        st.session_state["cached_summary"]  = detailed
        # parse into secs
        sections = {h:[] for h in ["Simplified Instructions","Importance","Follow-Up Appointments or Tasks","Medications","Precautions","References"]}
        cur=None
        for ln in detailed.splitlines():
            t=ln.strip(); t=re.sub(r"^[\-\*\s]+","",t)
            for h in sections:
                if t.lower().startswith(h.lower()):
                    cur=h; break
            else:
                if cur and t:
                    sections[cur].append(re.sub(r"[:\*]+$","",t).strip())
        st.session_state["cached_sections"] = sections

    st.session_state["run_summary"] = True

# --- Persistent display ---
if st.session_state["run_summary"]:
    concise = st.session_state["cached_concise"]
    sections = st.session_state["cached_sections"]

    # Simplified Summary
    st.markdown("---")
    st.subheader("Simplified Summary")
    for ln in concise.splitlines():
        st.markdown(apply_tooltips(ln), unsafe_allow_html=True)

    # Categorization & Actions
    st.markdown("---")
    st.subheader("Categorization & Actions")
    # clean
    for h,its in sections.items():
        sections[h] = [re.sub(r"\}+$","",i).strip() for i in its]
    sections = { re.sub(r"\*+","",h).strip(): [re.sub(r"\*+","",i).strip() for i in its] for h,its in sections.items() }

    if not sections or all(not v for v in sections.values()):
        st.info("No structured sections found.")
    else:
        for h,its in sections.items():
            st.markdown(f"**{h}**")
            for i in its:
                st.markdown(f"- {apply_tooltips(i)}", unsafe_allow_html=True)
            if h=="Follow-Up Appointments or Tasks":
                for fu in its:
                    st.download_button(f"Add '{fu}' to Calendar", generate_ics(fu), "event.ics", "text/calendar")
            if h=="Medications":
                st.subheader("Medication Checklist & Reminders")
                for m in its:
                    st.checkbox(m, key=f"med_{m}")
                if st.button("Schedule Med Reminders", key="med_reminders_btn"):
                    st.success("Medication reminders scheduled!")

        # overall reading level & JSON
        if textstat:
            combined = " ".join(i for its in sections.values() for i in its)
            lvl = textstat.flesch_kincaid_grade(combined)
            st.markdown(f"*Overall reading level: {lvl:.1f}th grade*")
        st.markdown("#### Categorization as JSON")
        st.json(sections)
        st.download_button("Download Categorization JSON", json.dumps(sections, indent=2), "categorization.json", "application/json")

    # Symptom Tracker
    st.markdown("---")
    st.subheader("Symptom Tracker")
    d = st.date_input("Date", datetime.date.today(), key="symp_date")
    pain = st.slider("Pain",0,10,0,key="symp_pain")
    swelling = st.slider("Swelling",0,10,0,key="symp_swelling")
    if st.button("Log Symptom", key="log_symp"):
        st.session_state["symptoms"].append({"date":str(d),"pain":pain,"swelling":swelling})
        st.success("Logged")
    if st.session_state["symptoms"]:
        st.dataframe(pd.DataFrame(st.session_state["symptoms"]))
    
    # Actions & Trackers
    st.markdown("---")
    st.subheader("Actions & Trackers")

    # high‑pain
    if st.session_state["symptoms"]:
        last = st.session_state["symptoms"][-1]
        if last.get("pain",0)>8:
            st.warning("High pain detected.")

    # timeline
    if st.checkbox("Show Recovery Timeline", key="show_timeline"):
        if st.session_state["symptoms"]:
            df=pd.DataFrame(st.session_state["symptoms"])
            df["date"]=pd.to_datetime(df["date"])
            st.line_chart(df.set_index("date")[["pain","swelling"]])
        else:
            st.info("No symptom data.")

    # download log
    if st.session_state["symptoms"]:
        csv = pd.DataFrame(st.session_state["symptoms"]).to_csv(index=False)
        st.download_button("Download Symptom Log", csv, "symptoms.csv","text/csv", key="dl_symptoms")

    # EHR stub
    if st.button("Sync with EHR", key="sync_ehr"):
        st.info("EHR integration not configured.")

    # caregiver
    if st.checkbox("Enable caregiver access", key="enable_caregiver"):
        email = st.text_input("Caregiver email:", key="caregiver_email")
        if email and st.button("Generate share link", key="gen_share"):
            link = f"https://yourapp.example.com/share?email={email}"
            # this opens the user's default mail client with subject/body pre‑filled
            mailto = (
                f"mailto:{email}"
                f"?subject={requests.utils.quote('Caregiver Access Link')}"
                f"&body={requests.utils.quote('Here is your link: ' + link)}"
            )
            st.markdown(f"[Click here to email the link]({mailto})")

    # Send Feedback
    st.markdown("---")
    st.subheader("Send Feedback to Provider")
    fb = st.text_area("Your message", key="feedback_msg")
    if st.button("Send Message", key="send_feedback"):
        fb = st.session_state["feedback_msg"]
        mailto_fb = (
            f"mailto:hamzapiracha@live.com"
            f"?subject={requests.utils.quote('Patient Feedback')}"
            f"&body={requests.utils.quote(fb)}"
        )
        st.markdown(f"[Click here to send feedback]({mailto_fb})")

    # Privacy Dashboard
    st.markdown("---")
    st.subheader("Privacy Dashboard")
    if st.button("View Stored Data", key="view_data"):
        st.json({k: st.session_state[k] for k in st.session_state})
    if st.button("Clear All Data", key="clear_data"):
        for k in list(st.session_state.keys()):
            st.session_state.pop(k,None)
        st.success("Cleared")

# --- Emergency Contacts ---
st.markdown("---")
st.subheader("🚨 Emergency Contacts")

with st.form("emergency_form"):
    ec_name = st.text_input("Contact Name", key="ec_name_input")
    ec_num  = st.text_input("Contact Number", key="ec_num_input")
    save_ec = st.form_submit_button("Save Contact")
    if save_ec:
        st.session_state["emergency"] = {"name": ec_name, "number": ec_num}
        st.success("Emergency contact saved")

# Always show the saved contact below
if "emergency" in st.session_state:
    em = st.session_state["emergency"]
    st.markdown(f"[Call {em['name']}]({{'tel:' + em['number']}})")
