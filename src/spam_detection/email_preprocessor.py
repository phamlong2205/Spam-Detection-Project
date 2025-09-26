# src/spam_detection/email_preprocessor.py
import re
from bs4 import BeautifulSoup

__all__ = [
    "strip_html", "remove_quoted_reply",
    "remove_emails_and_links", "normalize_whitespace",
    "clean_email_text"
]

def strip_html(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    return BeautifulSoup(text, "html.parser").get_text(separator="\n")

def remove_quoted_reply(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    lines = text.splitlines()
    out = []
    for ln in lines:
        if ln.strip().startswith(">"):
            continue
        if re.match(r"^On .+ wrote:$", ln.strip()):
            break
        out.append(ln)
    return "\n".join(out)

def remove_emails_and_links(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " EMAIL ", text)
    text = re.sub(r"(https?://|www\.)\S+", " URL ", text)
    return text

def normalize_whitespace(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()

def clean_email_text(raw: str) -> str:
    """
    Lightweight, email-specific pass done BEFORE your SMS/text normalizer.
    Keep it conservative so downstream tokenization & TF-IDF behave like SMS.
    """
    x = strip_html(raw)
    x = remove_quoted_reply(x)
    x = remove_emails_and_links(x)
    x = normalize_whitespace(x)
    return x
