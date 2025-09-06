
import os
import re
import imaplib
import email
from email.header import decode_header
import pandas as pd
import streamlit as st
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# -------------- Helpers --------------
def decode_maybe(value):
    if isinstance(value, bytes):
        try:
            return value.decode()
        except Exception:
            return value.decode(errors="ignore")
    return value

def parse_email_message(msg):
    subject, encoding = decode_header(msg.get("Subject"))[0]
    if isinstance(subject, bytes):
        try:
            subject = subject.decode(encoding or "utf-8", errors="ignore")
        except Exception:
            subject = subject.decode(errors="ignore")
    from_ = msg.get("From", "")
    date_ = msg.get("Date", "")
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            disp = str(part.get("Content-Disposition"))
            if ctype == "text/plain" and "attachment" not in disp:
                try:
                    body += part.get_payload(decode=True).decode(errors="ignore")
                except Exception:
                    pass
    else:
        try:
            body = msg.get_payload(decode=True).decode(errors="ignore")
        except Exception:
            body = str(msg.get_payload())

    return {
        "sender": from_,
        "subject": subject or "",
        "body": body or "",
        "date": date_
    }

def sentiment_label(text):
    positive_words = set("appreciate great thanks grateful love excellent awesome fantastic resolved happy pleased".split())
    negative_words = set("angry frustrated disappointed terrible bad poor delay delayed late broken cannot can't won't error failed failure down urgent immediately asap critical escalate unacceptable".split())
    tokens = re.findall(r"\b[\w']+\b", text.lower())
    pos = sum(1 for t in tokens if t in positive_words)
    neg = sum(1 for t in tokens if t in negative_words)
    if pos > neg:
        return "Positive"
    elif neg > pos:
        return "Negative"
    else:
        return "Neutral"

def priority_label(text):
    urgent_keywords = r"(immediately|urgent|asap|cannot access|can.?t access|down|critical|production|prod|escalate|deadline|blocked|failure|severe)"
    if re.search(urgent_keywords, text, flags=re.I):
        return "Urgent"
    return "Not urgent"

def extract_contacts(text):
    phone_regex = re.compile(r"(?:(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{4})")
    email_regex = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    phones = list(set(phone_regex.findall(text or "")))
    emails = list(set(email_regex.findall(text or "")))
    return ", ".join(phones), ", ".join(emails)

def extract_requirements(text):
    req_sentences = []
    for sent in re.split(r"(?<=[.!?])\s+", text.strip()):
        if re.search(r"\b(need|want|request|require|issue|problem|can't|cannot|unable|access|error|refund|cancel|upgrade|downgrade|exchange)\b", sent, flags=re.I):
            req_sentences.append(sent)
    return " ".join(req_sentences)[:500]

def extract_product(text):
    m = re.search(r"(product|plan|subscription|order|module|feature)\s*[:#-]?\s*([A-Za-z0-9\-\._]+)", text or "", flags=re.I)
    if m:
        return m.group(2)
    tokens = re.findall(r"\b[A-Z][a-zA-Z0-9\-]{2,}\b", text or "")
    common_excludes = {"I", "We", "Thanks", "Regards", "Hello", "Hi"}
    tokens = [t for t in tokens if t not in common_excludes]
    if tokens:
        return pd.Series(tokens).value_counts().index[0]
    return ""

def generate_reply(sender, subject, body, sentiment, priority, product, req_summary):
    greeting = "Hi there,"
    if isinstance(sender, str) and "@" in sender:
        name_guess = sender.split("@")[0].replace(".", " ").replace("_", " ").title()
        greeting = f"Hi {name_guess},"

    if sentiment == "Negative":
        empathy = "I'm sorry for the trouble you're facing. I understand how frustrating this can be, and I'm here to help. "
    elif sentiment == "Neutral":
        empathy = "Thanks for reaching out. I'd be happy to help. "
    else:
        empathy = "Thanks for the positive note—glad you contacted us. "

    product_line = f"We see this is related to {product}. " if product else ""
    urgency_line = "We've prioritized this request and will address it right away. " if priority == "Urgent" else ""
    request_line = f"From your message, I understand: {req_summary} " if req_summary else ""
    next_steps = "Could you confirm any recent changes and share a screenshot of the issue? Meanwhile, I'm checking logs and our knowledge base for similar reports. "
    closing = "We'll keep you posted on progress. If there's anything else you want to add, just reply to this email.\n\nBest regards,\nSupport Team"

    return f"""{greeting}

{empathy}{product_line}{urgency_line}{request_line}{next_steps}{closing}
"""

# -------------- Streamlit App --------------
st.set_page_config(page_title="AI-Powered Communication Assistant", layout="wide")
st.title("AI-Powered Communication Assistant")

st.sidebar.header("Data Source")
mode = st.sidebar.radio("Choose input mode", ["Upload CSV", "IMAP (demo)"])

if mode == "Upload CSV":
    file = st.sidebar.file_uploader("Upload a CSV with columns like sender, subject, body, date", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
elif mode == "IMAP (demo)":
    st.sidebar.write("Connect to an IMAP inbox (e.g., Gmail, Outlook).")
    host = st.sidebar.text_input("IMAP Host", value="imap.gmail.com")
    user = st.sidebar.text_input("Email / Username")
    password = st.sidebar.text_input("Password / App Password", type="password")
    folder = st.sidebar.text_input("Folder", value="INBOX")
    fetch_btn = st.sidebar.button("Fetch Emails")

    df = None
    if fetch_btn and host and user and password:
        try:
            M = imaplib.IMAP4_SSL(host)
            M.login(user, password)
            M.select(folder)
            status, data = M.search(None, "ALL")
            ids = data[0].split()[-100:]  # limit
            rows = []
            for i in ids:
                typ, msg_data = M.fetch(i, "(RFC822)")
                msg = email.message_from_bytes(msg_data[0][1])
                rows.append(parse_email_message(msg))
            df = pd.DataFrame(rows)
            M.logout()
        except Exception as e:
            st.error(f"IMAP error: {e}")

# If no df yet, show instructions
if "df" not in locals():
    st.info("Upload a CSV or connect via IMAP to get started.")
else:
    if df is not None and not df.empty:
        # Normalize columns
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        sender_col = next((c for c in df.columns if c in ["from", "sender", "sender_email", "email", "from_email"]), df.columns[0])
        subject_col = next((c for c in df.columns if "subject" in c), df.columns[1] if len(df.columns)>1 else df.columns[0])
        body_col = next((c for c in df.columns if c in ["body", "message", "email_body", "content"]), df.columns[2] if len(df.columns)>2 else df.columns[0])
        date_col = next((c for c in df.columns if "date" in c or "time" in c), None)

        # Filter support emails
        support_mask = df[subject_col].astype(str).str.contains(r"(support|query|request|help)", case=False, na=False) | \
                       df.get(body_col, pd.Series([""]*len(df))).astype(str).str.contains(r"(support|query|request|help)", case=False, na=False)
        sdf = df[support_mask].copy()

        # Enrich
        sdf["sentiment"] = (sdf[subject_col].astype(str) + " " + sdf[body_col].astype(str)).apply(sentiment_label)
        sdf["priority"] = (sdf[subject_col].astype(str) + " " + sdf[body_col].astype(str)).apply(priority_label)
        phones_emails = sdf[body_col].astype(str).apply(extract_contacts)
        sdf["phones"] = phones_emails.apply(lambda x: x[0])
        sdf["emails"] = phones_emails.apply(lambda x: x[1])
        sdf["requirements"] = sdf[body_col].astype(str).apply(extract_requirements)
        sdf["product"] = (sdf[subject_col].astype(str) + " " + sdf[body_col].astype(str)).apply(extract_product)
        sdf["draft_reply"] = sdf.apply(lambda r: generate_reply(r.get(sender_col,""), r.get(subject_col,""), r.get(body_col,""), r["sentiment"], r["priority"], r["product"], r["requirements"]), axis=1)
        # Priority queue
        sdf["__rank"] = np.where(sdf["priority"]=="Urgent", 0, 1)
        # Date coercion
        if date_col:
            try:
                sdf["__date"] = pd.to_datetime(sdf[date_col], errors="coerce")
            except Exception:
                sdf["__date"] = pd.NaT
        else:
            sdf["__date"] = pd.NaT
        sdf = sdf.sort_values(by=["__rank","__date"], ascending=[True, False])

        # Analytics
        st.subheader("Analytics")
        col1, col2, col3, col4 = st.columns(4)
        total = len(sdf)
        urgent = int((sdf["priority"]=="Urgent").sum())
        positive = int((sdf["sentiment"]=="Positive").sum())
        negative = int((sdf["sentiment"]=="Negative").sum())
        with col1: st.metric("Total Support Emails", total)
        with col2: st.metric("Urgent", urgent)
        with col3: st.metric("Positive", positive)
        with col4: st.metric("Negative", negative)

        # Charts
        st.subheader("Charts")
        fig1, ax1 = plt.subplots()
        sdf["sentiment"].value_counts().plot(kind="bar", ax=ax1)
        ax1.set_title("Sentiment Distribution")
        ax1.set_xlabel("Sentiment")
        ax1.set_ylabel("Count")
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots()
        sdf["priority"].value_counts().plot(kind="bar", ax=ax2)
        ax2.set_title("Priority Distribution")
        ax2.set_xlabel("Priority")
        ax2.set_ylabel("Count")
        st.pyplot(fig2)

        # Table with details + editable response
        st.subheader("Support Emails")
        for idx, row in sdf.iterrows():
            with st.expander(f"{row.get(subject_col,'(no subject)')} — {row.get(sender_col,'(no sender)')}"):
                st.write(f"**Date:** {row.get(date_col, '')}")
                st.write(f"**Priority:** {row['priority']} | **Sentiment:** {row['sentiment']} | **Product:** {row.get('product','')}")
                st.write("**Body**")
                st.text(row.get(body_col, ""))
                st.write(f"**Phones:** {row.get('phones','')} | **Emails:** {row.get('emails','')}")
                st.write(f"**Requirements:** {row.get('requirements','')}")
                draft = st.text_area("Draft Reply", value=row.get("draft_reply",""), height=200, key=f"draft_{idx}")
                st.button("Mark as Resolved", key=f"resolve_{idx}")

        # Export
        st.download_button("Download Enriched CSV", data=sdf.to_csv(index=False).encode("utf-8"), file_name="enriched_support_emails.csv", mime="text/csv")

    else:
        st.info("No data found.")

