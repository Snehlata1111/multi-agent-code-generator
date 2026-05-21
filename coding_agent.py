import os
import json
import re
import streamlit as st
from huggingface_hub import InferenceClient
from pathlib import Path
from pygments import highlight
from pygments.lexers import get_lexer_by_name, guess_lexer, TextLexer
from pygments.formatters import HtmlFormatter

# Load .env file if present
DOTENV_PATH = Path(__file__).parent / ".env"
if DOTENV_PATH.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=DOTENV_PATH)
    except ImportError:
        pass

# =========================
# PAGE CONFIG — must be first
# =========================

st.set_page_config(
    page_title="IntelliDev Agent",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CUSTOM CSS
# =========================

st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Main background ── */
.stApp {
    background: #0d1117;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #161b22 !important;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] * {
    color: #e6edf3 !important;
}

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #161b22 0%, #0d1117 60%, #1a1f2e 100%);
    border: 1px solid #21262d;
    border-radius: 16px;
    padding: 36px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #4a9eff, #a371f7, #3fb950);
    border-radius: 16px 16px 0 0;
}
.hero-title {
    font-size: 2rem;
    font-weight: 700;
    color: #e6edf3;
    margin: 0 0 8px 0;
    letter-spacing: -0.5px;
}
.hero-sub {
    font-size: 1rem;
    color: #8b949e;
    margin: 0;
}
.badge-row {
    display: flex;
    gap: 8px;
    margin-top: 16px;
    flex-wrap: wrap;
}
.badge {
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.badge-blue  { background: #1a2744; color: #4a9eff; border: 1px solid #4a9eff44; }
.badge-green { background: #1a3a2a; color: #3fb950; border: 1px solid #3fb95044; }
.badge-purple{ background: #2a1a44; color: #a371f7; border: 1px solid #a371f744; }
.badge-orange{ background: #3a2a10; color: #ffa657; border: 1px solid #ffa65744; }

/* ── Pipeline steps ── */
.pipeline {
    display: flex;
    align-items: center;
    gap: 0;
    margin-bottom: 28px;
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 16px 24px;
    overflow-x: auto;
}
.step {
    display: flex;
    align-items: center;
    gap: 8px;
    white-space: nowrap;
}
.step-circle {
    width: 32px; height: 32px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 13px; font-weight: 700;
}
.step-active   { background: #4a9eff22; border: 2px solid #4a9eff; color: #4a9eff; }
.step-done     { background: #3fb95022; border: 2px solid #3fb950; color: #3fb950; }
.step-inactive { background: #21262d;   border: 2px solid #30363d; color: #484f58; }
.step-label { font-size: 12px; font-weight: 600; color: #8b949e; }
.step-label.active { color: #e6edf3; }
.step-arrow { color: #30363d; margin: 0 12px; font-size: 16px; }

/* ── Cards ── */
.card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 20px;
}
.card-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid #21262d;
}
.card-icon {
    width: 36px; height: 36px;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px;
}
.icon-blue   { background: #1a2744; }
.icon-green  { background: #1a3a2a; }
.icon-purple { background: #2a1a44; }
.icon-orange { background: #3a2a10; }
.card-title {
    font-size: 14px;
    font-weight: 700;
    color: #e6edf3;
    margin: 0;
}
.card-subtitle {
    font-size: 11px;
    color: #8b949e;
    margin: 0;
}

/* ── Problem summary box ── */
.problem-box {
    background: #0d1117;
    border: 1px solid #21262d;
    border-left: 3px solid #4a9eff;
    border-radius: 8px;
    padding: 14px 16px;
    font-size: 14px;
    color: #c9d1d9;
    line-height: 1.6;
    margin-bottom: 16px;
}

/* ── Subtask pills ── */
.subtask-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
    margin-bottom: 4px;
}
.subtask-item {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 13px;
    color: #c9d1d9;
    display: flex;
    align-items: center;
    gap: 8px;
}
.subtask-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #3fb950;
    flex-shrink: 0;
}

/* ── Tech stack chips ── */
.tech-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 4px;
}
.tech-chip {
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 12px;
    color: #8b949e;
    font-family: 'JetBrains Mono', monospace;
}

/* ── File header ── */
.file-header {
    background: #161b22;
    border: 1px solid #21262d;
    border-bottom: none;
    border-radius: 10px 10px 0 0;
    padding: 10px 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.file-dot { width: 10px; height: 10px; border-radius: 50%; }
.dot-red    { background: #f85149; }
.dot-yellow { background: #d29922; }
.dot-green  { background: #3fb950; }
.file-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #8b949e;
    margin-left: 8px;
}

/* ── Section divider ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #21262d, transparent);
    margin: 24px 0;
}

/* ── Sidebar elements ── */
.sidebar-logo {
    text-align: center;
    padding: 20px 0 24px 0;
    border-bottom: 1px solid #21262d;
    margin-bottom: 20px;
}
.sidebar-logo-text {
    font-size: 20px;
    font-weight: 700;
    color: #e6edf3;
}
.sidebar-logo-sub {
    font-size: 11px;
    color: #8b949e;
    margin-top: 4px;
}
.model-badge {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 10px 12px;
    margin-top: 16px;
    font-size: 11px;
    color: #8b949e;
}
.model-badge span {
    color: #4a9eff;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
}

/* ── Streamlit widget overrides ── */
.stTextArea textarea {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    color: #e6edf3 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
}
.stTextArea textarea:focus {
    border-color: #4a9eff !important;
    box-shadow: 0 0 0 3px #4a9eff22 !important;
}
.stButton > button {
    background: linear-gradient(135deg, #1a2744, #0f1a30) !important;
    border: 1px solid #4a9eff !important;
    color: #4a9eff !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 8px 20px !important;
    transition: all 0.2s !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: #4a9eff !important;
    color: #0d1117 !important;
}
.stSelectbox > div > div {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    color: #e6edf3 !important;
}
div[data-testid="stMarkdownContainer"] p {
    color: #c9d1d9;
}
div[data-testid="stMarkdownContainer"] li {
    color: #c9d1d9;
}
div[data-testid="stMarkdownContainer"] h1,
div[data-testid="stMarkdownContainer"] h2,
div[data-testid="stMarkdownContainer"] h3,
div[data-testid="stMarkdownContainer"] h4 {
    color: #e6edf3;
}
div[data-testid="stMarkdownContainer"] code {
    background: #21262d;
    color: #a371f7;
    padding: 2px 6px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
}
/* Fix code blocks inside review output */
div[data-testid="stMarkdownContainer"] pre {
    background: #1e1e1e !important;
    border: 1px solid #21262d !important;
    border-radius: 8px !important;
}
div[data-testid="stMarkdownContainer"] pre code {
    background: transparent !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
}
.stCodeBlock pre, [data-testid="stCode"] pre {
    background: #1e1e1e !important;
}
/* Force all code text to be bright */
.stCodeBlock code, .stCodeBlock span,
[data-testid="stCode"] code, [data-testid="stCode"] span,
div[data-testid="stMarkdownContainer"] pre span {
    color: #e6edf3 !important;
    opacity: 1 !important;
    filter: none !important;
}
</style>
""", unsafe_allow_html=True)

# =========================
# SYNTAX HIGHLIGHTED CODE RENDERER
# =========================

# VS Code Dark+ color palette via Pygments
VSCODE_CSS = """
.highlight { background: #1e1e1e; border-radius: 8px; padding: 16px 20px; overflow-x: auto; font-family: 'JetBrains Mono', 'Consolas', monospace; font-size: 13px; line-height: 1.6; border: 1px solid #21262d; white-space: pre; }
.highlight .c,  .highlight .c1, .highlight .cm { color: #6a9955; font-style: italic; } /* comments - green */
.highlight .k,  .highlight .kd, .highlight .kn, .highlight .kp, .highlight .kr { color: #569cd6; font-weight: bold; } /* keywords - blue */
.highlight .kt { color: #4ec9b0; } /* type keywords - teal */
.highlight .s,  .highlight .s1, .highlight .s2, .highlight .sb, .highlight .sc { color: #ce9178; } /* strings - orange */
.highlight .mi, .highlight .mf, .highlight .mh { color: #b5cea8; } /* numbers - light green */
.highlight .n  { color: #d4d4d4; } /* names - light grey */
.highlight .na { color: #9cdcfe; } /* attribute names - light blue */
.highlight .nb { color: #dcdcaa; } /* builtins - yellow */
.highlight .nc { color: #4ec9b0; } /* class names - teal */
.highlight .nf { color: #dcdcaa; } /* function names - yellow */
.highlight .nn { color: #d4d4d4; } /* namespace */
.highlight .o  { color: #d4d4d4; } /* operators */
.highlight .p  { color: #d4d4d4; } /* punctuation */
.highlight .bp { color: #569cd6; } /* self/cls */
.highlight .vi { color: #9cdcfe; } /* instance vars */
.highlight .err { color: #f44747; background: none; } /* errors - red */
"""

def render_code(code, ext):
    try:
        lexer = get_lexer_by_name(ext, stripall=True)
    except Exception:
        try:
            lexer = guess_lexer(code)
        except Exception:
            lexer = TextLexer()
    formatter = HtmlFormatter(style="monokai", noclasses=False, nowrap=False)
    highlighted = highlight(code, lexer, formatter)
    st.markdown(f"<style>{VSCODE_CSS}</style>{highlighted}", unsafe_allow_html=True)


# =========================
# HUGGING FACE API CHECK
# =========================

HF_API_KEY = os.getenv("HF_API_KEY")

if not HF_API_KEY:
    st.markdown("""
    <div class="card" style="border-left: 3px solid #f85149; margin-top: 40px;">
        <div class="card-header">
            <div class="card-icon icon-orange">🔑</div>
            <div>
                <p class="card-title">API Key Missing</p>
                <p class="card-subtitle">HF_API_KEY environment variable not set</p>
            </div>
        </div>
        <p style="color:#8b949e; font-size:13px; margin:0;">
            Set your Hugging Face API key in a <code>.env</code> file or as an environment variable.<br>
            Get a free token at <a href="https://huggingface.co/settings/tokens" style="color:#4a9eff;">huggingface.co/settings/tokens</a>
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

client = InferenceClient(api_key=HF_API_KEY)

# =========================
# JSON EXTRACTION HELPER
# =========================

def extract_json(text):
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    candidate = match.group(0)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    fixed = re.sub(
        r'("(?:[^"\\]|\\.)*")',
        lambda m: m.group(0).replace("\n", "\\n").replace("\r", ""),
        candidate,
        flags=re.DOTALL
    )
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        return None

# =========================
# AGENT FUNCTIONS
# =========================

def ask_model(prompt):
    try:
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-72B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"API request failed: {str(e)}")
        return None

def generate_plan(problem_input):
    prompt = f"""You are an expert software architect. Analyze the project idea below and respond with ONLY a valid JSON object — no explanation, no markdown, no extra text.

Project: {problem_input}

Respond with exactly this JSON structure:
{{
    "problem": "one sentence project summary",
    "subtasks": ["task 1", "task 2", "task 3"],
    "tech_stack": ["Technology1", "Technology2", "Technology3"]
}}"""
    response = ask_model(prompt)
    if not response:
        return {"problem": problem_input, "subtasks": ["Frontend Development", "Backend Development", "Database Setup"], "tech_stack": ["React", "Python", "FastAPI"]}
    result = extract_json(response)
    if result:
        return result
    return {"problem": problem_input, "subtasks": ["Frontend Development", "Backend Development", "Database Setup"], "tech_stack": ["React", "Python", "FastAPI"]}

def generate_code(task, tech_stack):
    tech = ", ".join(tech_stack) if isinstance(tech_stack, list) else tech_stack
    prompt = f"""You are a senior software engineer. Generate code for the task below.

Task: {task}
Tech Stack: {tech}

Respond with ONLY a valid JSON object. No text before or after. No markdown fences around the JSON itself.
IMPORTANT: In the "code" field, use actual newline characters (press Enter) for each new line. Do NOT write \\n as text.

{{
    "filename": "appropriate_filename.py",
    "code": "first line
second line
third line",
    "explanation": "brief explanation"
}}"""
    response = ask_model(prompt)
    if not response:
        return {"filename": "example.py", "code": "# Code generation failed", "explanation": "API request failed."}

    # Try JSON extraction first
    result = extract_json(response)
    if result and "code" in result and len(result["code"]) > 10:
        code = result["code"]
        # Restore escaped newlines
        code = code.replace("\\n", "\n").replace("\\t", "\t")
        result["code"] = code
        return result

    # Fallback 1: extract ```lang ... ``` block
    code_match = re.search(r"```(?:\w+)?\n(.*?)```", response, re.DOTALL)
    if code_match:
        code = code_match.group(1).strip()
        fname_match = re.search(r'"filename"\s*:\s*"([^"]+)"', response)
        fname = fname_match.group(1) if fname_match else "generated.py"
        exp_match = re.search(r'"explanation"\s*:\s*"([^"]+)"', response)
        exp = exp_match.group(1) if exp_match else "Code extracted from response."
        return {"filename": fname, "code": code, "explanation": exp}

    # Fallback 2: treat entire response as code if it looks like code
    if any(kw in response for kw in ["import ", "def ", "class ", "const ", "function ", "var ", "let "]):
        return {"filename": "generated.py", "code": response.strip(), "explanation": "Code extracted from response."}

    return {"filename": "generated.py", "code": "# Could not extract code\n# Raw response:\n# " + response[:200], "explanation": "Extraction failed."}

def review_code(code):
    prompt = f"""You are a senior code reviewer. Review this code and provide:
- Bug analysis
- Security improvements
- Performance optimizations
- Best practices

Code:
{code}"""
    return ask_model(prompt)

# =========================
# SESSION STATE
# =========================

for key in ["code_plan", "generated_code", "code_review"]:
    if key not in st.session_state:
        st.session_state[key] = None

# =========================
# SIDEBAR
# =========================

with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div style="font-size:32px; margin-bottom:8px;">💻</div>
        <div class="sidebar-logo-text">IntelliDev Agent</div>
        <div class="sidebar-logo-sub">AI-Powered Development Pipeline</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Describe your project**")
    problem_input = st.text_area(
        label="problem",
        placeholder="e.g. Build a REST API for a todo app with user auth and CRUD operations...",
        height=140,
        label_visibility="collapsed"
    )

    plan_clicked = st.button("🧠  Plan & Build", use_container_width=True)

    if plan_clicked and problem_input.strip():
        with st.spinner("Planning architecture..."):
            st.session_state.code_plan = generate_plan(problem_input)
            st.session_state.generated_code = None
            st.session_state.code_review = None

    st.markdown("""
    <div class="model-badge">
        🤖 Model: <span>Qwen2.5-72B-Instruct</span><br>
        🔌 Provider: <span>Hugging Face</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown("**Example prompts**")
    st.markdown("""
    <div style="font-size:11px; color:#8b949e; line-height:2;">
    • Build a REST API for a todo list<br>
    • Create a user auth system<br>
    • Real-time chat with WebSockets<br>
    • S3 file upload with DynamoDB
    </div>
    """, unsafe_allow_html=True)

# =========================
# MAIN CONTENT
# =========================

# Hero
st.markdown("""
<div class="hero">
    <h1 class="hero-title">💻 IntelliDev Agent</h1>
    <p class="hero-sub">Describe a feature → get an architecture plan → generate code → automated review</p>
    <div class="badge-row">
        <span class="badge badge-blue">Planner Agent</span>
        <span class="badge badge-purple">Code Generator</span>
        <span class="badge badge-orange">Reviewer Agent</span>
        <span class="badge badge-green">Qwen2.5-72B</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Pipeline progress indicator
step1 = "done" if st.session_state.code_plan else "active"
step2 = "done" if st.session_state.generated_code else ("active" if st.session_state.code_plan else "inactive")
step3 = "done" if st.session_state.code_review else ("active" if st.session_state.generated_code else "inactive")

def step_html(num, label, state):
    circle_class = f"step-{state}"
    label_class  = "active" if state in ("active", "done") else ""
    icon = "&#10003;" if state == "done" else str(num)
    return (
        '<div class="step">'
        f'<div class="step-circle {circle_class}">{icon}</div>'
        f'<span class="step-label {label_class}">{label}</span>'
        '</div>'
    )

st.markdown(f"""
<div class="pipeline">
    {step_html(1, "Plan", step1)}
    <span class="step-arrow">→</span>
    {step_html(2, "Generate", step2)}
    <span class="step-arrow">→</span>
    {step_html(3, "Review", step3)}
</div>
""", unsafe_allow_html=True)

# =========================
# PLAN OUTPUT
# =========================

if st.session_state.code_plan:
    plan = st.session_state.code_plan

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
        <div class="card">
            <div class="card-header">
                <div class="card-icon icon-blue">🧠</div>
                <div>
                    <p class="card-title">Project Analysis</p>
                    <p class="card-subtitle">Planner Agent Output</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f'<div class="problem-box">{plan["problem"]}</div>', unsafe_allow_html=True)

        subtasks_html = "".join([
            f'<div class="subtask-item"><div class="subtask-dot"></div>{task}</div>'
            for task in plan["subtasks"]
        ])
        st.markdown(f"""
        <p style="font-size:12px; font-weight:600; color:#8b949e; margin-bottom:8px; text-transform:uppercase; letter-spacing:1px;">Subtasks</p>
        <div class="subtask-list">{subtasks_html}</div>
        """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        chips_html = "".join([f'<span class="tech-chip">{t}</span>' for t in plan["tech_stack"]])
        st.markdown(f"""
        <div class="card">
            <div class="card-header">
                <div class="card-icon icon-green">🛠️</div>
                <div>
                    <p class="card-title">Tech Stack</p>
                    <p class="card-subtitle">Recommended technologies</p>
                </div>
            </div>
            <div class="tech-chips">{chips_html}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <div class="card-header">
            <div class="card-icon icon-purple">⚙️</div>
            <div>
                <p class="card-title">Code Generator</p>
                <p class="card-subtitle">Select a subtask to implement</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    selected_task = st.selectbox(
        "Subtask",
        plan["subtasks"],
        label_visibility="collapsed"
    )

    gen_clicked = st.button("⚙️  Generate Code", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if gen_clicked:
        with st.spinner("Generating implementation..."):
            st.session_state.generated_code = generate_code(selected_task, plan["tech_stack"])
            st.session_state.code_review = None

# =========================
# CODE OUTPUT
# =========================

if st.session_state.generated_code:
    result = st.session_state.generated_code
    ext = result["filename"].split(".")[-1]

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="file-header">
        <div class="file-dot dot-red"></div>
        <div class="file-dot dot-yellow"></div>
        <div class="file-dot dot-green"></div>
        <span class="file-name">{result['filename']}</span>
    </div>
    """, unsafe_allow_html=True)

    render_code(result["code"], ext)

    st.markdown(f"""
    <div class="card" style="margin-top:16px;">
        <div class="card-header">
            <div class="card-icon icon-blue">🧠</div>
            <div>
                <p class="card-title">Explanation</p>
                <p class="card-subtitle">What this code does</p>
            </div>
        </div>
        <p style="font-size:13px; color:#c9d1d9; line-height:1.7; margin:0;">{result['explanation']}</p>
    </div>
    """, unsafe_allow_html=True)

    review_clicked = st.button("🔍  Review Code", use_container_width=False)

    if review_clicked:
        with st.spinner("Reviewing code..."):
            st.session_state.code_review = review_code(result["code"])

# =========================
# REVIEW OUTPUT
# =========================

if st.session_state.code_review:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <div class="card-header">
            <div class="card-icon icon-orange">🔍</div>
            <div>
                <p class="card-title">Code Review</p>
                <p class="card-subtitle">Reviewer Agent Analysis</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.markdown(st.session_state.code_review)
    st.markdown("</div>", unsafe_allow_html=True)
