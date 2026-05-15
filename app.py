import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pypdf import PdfReader
import tempfile
import os

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="StudyBuddy RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #e8e8f0;
}

h1, h2, h3 {
    font-family: 'Sora', sans-serif;
    font-weight: 700;
}

.answer-box {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 1.5rem;
    margin-top: 1rem;
    font-size: 1rem;
    line-height: 1.7;
    backdrop-filter: blur(12px);
}

.chip {
    display: inline-block;
    background: rgba(120,100,255,0.25);
    border: 1px solid rgba(120,100,255,0.5);
    border-radius: 999px;
    padding: 2px 12px;
    font-size: 0.78rem;
    margin: 3px;
    font-family: 'JetBrains Mono', monospace;
}

.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #a89cff;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  GEMINI SETUP
# ─────────────────────────────────────────────
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    api_key = os.environ.get("GEMINI_API_KEY", "")

if not api_key:
    st.error("⚠️ No Gemini API key found. Add GEMINI_API_KEY to `.streamlit/secrets.toml`.")
    st.stop()

genai.configure(api_key=api_key)
llm = genai.GenerativeModel("gemini-1.5-flash")

# ─────────────────────────────────────────────
#  EMBEDDING MODEL (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model…")
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def extract_text(pdf_file) -> str:
    reader = PdfReader(pdf_file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80):
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def build_index(chunks):
    vecs = embedder.encode(chunks, show_progress_bar=False)
    idx = faiss.IndexFlatL2(vecs.shape[1])
    idx.add(np.array(vecs, dtype="float32"))
    return idx, vecs


def retrieve(query: str, chunks, index, k: int = 4):
    qvec = embedder.encode([query], show_progress_bar=False)
    _, idxs = index.search(np.array(qvec, dtype="float32"), k=k)
    return [chunks[i] for i in idxs[0] if i < len(chunks)]


def ask_llm(prompt: str) -> str:
    response = llm.generate_content(prompt)
    return response.text


# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
for key in ["chunks", "index", "history", "pdf_name"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key in ["chunks", "history"] else None

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📚 StudyBuddy RAG")
    st.markdown("---")

    uploaded = st.file_uploader(
        "Upload PDF Notes",
        type="pdf",
        accept_multiple_files=False,
        help="Upload any lecture notes, textbooks, or study material as PDF."
    )

    chunk_size = st.slider("Chunk size (chars)", 200, 1000, 500, 50,
                           help="Smaller = more precise, Larger = more context.")
    top_k = st.slider("Chunks to retrieve", 2, 8, 4,
                      help="How many text chunks to pass to the AI.")

    if uploaded:
        if st.session_state.pdf_name != uploaded.name:
            with st.spinner("Processing PDF…"):
                raw = extract_text(uploaded)
                chunks = chunk_text(raw, chunk_size=chunk_size)
                index, _ = build_index(chunks)
                st.session_state.chunks = chunks
                st.session_state.index = index
                st.session_state.pdf_name = uploaded.name
                st.session_state.history = []
            st.success(f"✅ {len(chunks)} chunks indexed!")

    st.markdown("---")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.history = []
        st.rerun()

    st.markdown("""
    <div style='font-size:0.75rem; color:#888; margin-top:1rem;'>
    Built with Gemini + FAISS + Streamlit
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MAIN AREA
# ─────────────────────────────────────────────
st.markdown("# 📚 StudyBuddy RAG Assistant")
st.markdown("*Ask questions, generate summaries, quizzes, and flashcards from your notes.*")
st.markdown("---")

if not st.session_state.chunks:
    st.info("👈 Upload a PDF from the sidebar to get started.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📝 Summary", "❓ Quiz", "🃏 Flashcards"])

# ── TAB 1: CHAT ──────────────────────────────
with tab1:
    st.markdown('<div class="section-header">Ask anything from your notes</div>', unsafe_allow_html=True)

    for msg in st.session_state.history:
        role_icon = "🧑" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"]):
            st.write(f"{role_icon} {msg['content']}")

    question = st.chat_input("Type your question…")
    if question:
        st.session_state.history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(f"🧑 {question}")

        context_chunks = retrieve(question, st.session_state.chunks, st.session_state.index, k=top_k)
        context = "\n\n".join(context_chunks)

        # Build conversation-aware prompt
        history_text = "\n".join(
            f"{'Student' if m['role']=='user' else 'StudyBuddy'}: {m['content']}"
            for m in st.session_state.history[-6:]
        )

        prompt = f"""You are StudyBuddy, a friendly and expert academic tutor.
Use ONLY the context below to answer the student's question clearly and concisely.
If the answer is not in the context, say so honestly.

--- CONTEXT FROM NOTES ---
{context}

--- CONVERSATION HISTORY ---
{history_text}

--- STUDENT'S QUESTION ---
{question}

Answer in clear, simple language. Use bullet points or numbered lists where helpful."""

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                answer = ask_llm(prompt)
            st.write(f"🤖 {answer}")
            st.session_state.history.append({"role": "assistant", "content": answer})

# ── TAB 2: SUMMARY ───────────────────────────
with tab2:
    st.markdown('<div class="section-header">Generate a Summary</div>', unsafe_allow_html=True)
    topic = st.text_input("Optional: focus on a specific topic (leave blank for full summary)")

    if st.button("📝 Generate Summary", use_container_width=True):
        query = topic if topic else "main concepts overview"
        chunks = retrieve(query, st.session_state.chunks, st.session_state.index, k=6)
        context = "\n\n".join(chunks)

        prompt = f"""Summarize the following study material clearly and concisely.
{"Focus on: " + topic if topic else "Cover all key concepts."}
Use headings, bullet points, and highlight important terms.

CONTENT:
{context}"""

        with st.spinner("Summarizing…"):
            summary = ask_llm(prompt)

        st.markdown('<div class="answer-box">' + summary.replace("\n", "<br>") + '</div>',
                    unsafe_allow_html=True)

# ── TAB 3: QUIZ ──────────────────────────────
with tab3:
    st.markdown('<div class="section-header">Quiz Generator</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        quiz_topic = st.text_input("Topic for quiz (optional)")
        num_q = st.selectbox("Number of questions", [3, 5, 8, 10], index=1)
    with col2:
        difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"])
        q_type = st.selectbox("Question type", ["Multiple Choice", "Short Answer", "True/False"])

    if st.button("❓ Generate Quiz", use_container_width=True):
        query = quiz_topic if quiz_topic else "key concepts"
        chunks = retrieve(query, st.session_state.chunks, st.session_state.index, k=6)
        context = "\n\n".join(chunks)

        prompt = f"""Create {num_q} {difficulty.lower()} {q_type} questions from the content below.
Format each question clearly with the correct answer at the end marked as ✅ Answer: ...

CONTENT:
{context}"""

        with st.spinner("Generating quiz…"):
            quiz = ask_llm(prompt)

        st.markdown('<div class="answer-box">' + quiz.replace("\n", "<br>") + '</div>',
                    unsafe_allow_html=True)

# ── TAB 4: FLASHCARDS ────────────────────────
with tab4:
    st.markdown('<div class="section-header">Flashcard Generator</div>', unsafe_allow_html=True)
    fc_topic = st.text_input("Topic for flashcards (optional)")
    num_fc = st.selectbox("Number of flashcards", [5, 10, 15, 20], index=1)

    if st.button("🃏 Generate Flashcards", use_container_width=True):
        query = fc_topic if fc_topic else "key terms and definitions"
        chunks = retrieve(query, st.session_state.chunks, st.session_state.index, k=6)
        context = "\n\n".join(chunks)

        prompt = f"""Create {num_fc} flashcards from the content below.
Format EXACTLY like this for each card:

FRONT: [Term or concept]
BACK: [Clear, concise explanation]
---

CONTENT:
{context}"""

        with st.spinner("Creating flashcards…"):
            raw = ask_llm(prompt)

        cards = [c.strip() for c in raw.split("---") if "FRONT:" in c and "BACK:" in c]

        if cards:
            for i, card in enumerate(cards, 1):
                lines = card.strip().splitlines()
                front = next((l.replace("FRONT:", "").strip() for l in lines if l.startswith("FRONT:")), "")
                back = next((l.replace("BACK:", "").strip() for l in lines if l.startswith("BACK:")), "")

                with st.expander(f"🃏 Card {i}: {front}"):
                    st.markdown(f"**Answer:** {back}")
        else:
            st.markdown('<div class="answer-box">' + raw.replace("\n", "<br>") + '</div>',
                        unsafe_allow_html=True)
