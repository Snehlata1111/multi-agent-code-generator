import os
import json
import re
import streamlit as st
from huggingface_hub import InferenceClient
from pathlib import Path

# Load .env file if present
DOTENV_PATH = Path(__file__).parent / ".env"
if DOTENV_PATH.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=DOTENV_PATH)
    except ImportError:
        pass

# =========================
# HUGGING FACE API CONFIGURATION
# =========================

HF_API_KEY = os.getenv("HF_API_KEY")

if not HF_API_KEY:
    st.set_page_config(
        page_title="💻 IntelliDev Agent",
        layout="wide"
    )
    st.title("💻 IntelliDev Agent")
    st.error(
        "Missing HF_API_KEY environment variable.\n"
        "Set HF_API_KEY before running the app."
    )
    st.stop()

client = InferenceClient(api_key=HF_API_KEY)

# =========================
# STREAMLIT PAGE CONFIG
# =========================

st.set_page_config(
    page_title="💻 IntelliDev Agent",
    layout="wide"
)

st.title("💻 IntelliDev Agent")

st.markdown(
    "Streamline your development with planning, coding, and reviewing agents."
)

# =========================
# JSON EXTRACTION HELPER
# =========================

def extract_json(text):
    """
    Robustly extract a JSON object from a model response that may contain
    markdown fences, extra prose, or unescaped newlines inside string values.
    """
    if not text:
        return None

    # Strip markdown code fences
    text = re.sub(r"```(?:json)?", "", text).strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find the outermost { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None

    candidate = match.group(0)

    # Try parsing as-is
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Last resort: replace literal newlines inside string values with \n
    # so json.loads can handle multi-line code blocks embedded in JSON
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
# HUGGING FACE HELPER FUNCTION
# =========================

def ask_gemini(prompt):

    try:
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-72B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.7
        )
        return response.choices[0].message.content

    except Exception as e:
        error_text = str(e)
        st.error(
            f"Hugging Face request failed. Check your API key and internet connection.\n{error_text}"
        )
        return None

# =========================
# PLANNER AGENT
# =========================

def generate_plan(problem_input):

    prompt = f"""You are an expert software architect. Analyze the project idea below and respond with ONLY a valid JSON object — no explanation, no markdown, no extra text.

Project: {problem_input}

Respond with exactly this JSON structure:
{{
    "problem": "one sentence project summary",
    "subtasks": ["task 1", "task 2", "task 3"],
    "tech_stack": ["Technology1", "Technology2", "Technology3"]
}}"""

    response = ask_gemini(prompt)

    if not response:
        return {
            "problem": problem_input,
            "subtasks": ["Frontend Development", "Backend Development", "Database Setup"],
            "tech_stack": ["React", "Python", "FastAPI"]
        }

    result = extract_json(response)

    if result:
        return result

    st.warning("Planner returned unexpected format. Using default structure.")
    return {
        "problem": problem_input,
        "subtasks": ["Frontend Development", "Backend Development", "Database Setup"],
        "tech_stack": ["React", "Python", "FastAPI"]
    }

# =========================
# CODE GENERATOR AGENT
# =========================

def generate_code(task, tech_stack):

    tech = ", ".join(tech_stack) if isinstance(tech_stack, list) else tech_stack

    prompt = f"""You are a senior software engineer. Generate production-ready code for the task below.

Task: {task}
Tech Stack: {tech}

Respond with ONLY a valid JSON object — no explanation outside the JSON, no markdown fences.
Use \\n for newlines inside the "code" string value.

{{
    "filename": "appropriate_filename.py",
    "code": "full code here with \\n for line breaks",
    "explanation": "brief explanation of what the code does"
}}"""

    response = ask_gemini(prompt)

    if not response:
        return {
            "filename": "example.py",
            "code": "# Code generation failed",
            "explanation": "API request failed. Check your API key and internet connection."
        }

    result = extract_json(response)

    if result:
        # Restore actual newlines in code if they were escaped
        if "code" in result:
            result["code"] = result["code"].replace("\\n", "\n")
        return result

    # Fallback: try to extract just the code block from the raw response
    code_match = re.search(r"```(?:\w+)?\n(.*?)```", response, re.DOTALL)
    code = code_match.group(1).strip() if code_match else "# Could not extract code"

    return {
        "filename": "generated.py",
        "code": code,
        "explanation": "JSON parsing failed — code extracted directly from response."
    }

# =========================
# REVIEWER AGENT
# =========================

def review_code(code):

    prompt = f"""
You are a senior code reviewer.

Review this code:

{code}

Provide:
- bug analysis
- optimization suggestions
- security improvements
- best practices
"""

    return ask_gemini(prompt)

# =========================
# SESSION STATE
# =========================

if "code_plan" not in st.session_state:
    st.session_state.code_plan = None

if "generated_code" not in st.session_state:
    st.session_state.generated_code = None

if "code_review" not in st.session_state:
    st.session_state.code_review = None

# =========================
# SIDEBAR INPUT
# =========================

with st.sidebar:

    st.header("📥 Problem Input")

    problem_input = st.text_area(
        "Describe your coding problem or feature request:"
    )

    if st.button("Plan & Build"):

        with st.spinner("🧠 Planning project architecture..."):

            st.session_state.code_plan = generate_plan(problem_input)

# =========================
# DISPLAY PLAN
# =========================

if st.session_state.code_plan:

    plan = st.session_state.code_plan

    st.markdown("## 📋 Project Plan")

    st.markdown("### Problem")
    st.markdown(plan["problem"])

    st.markdown("### 🧩 Subtasks")

    for task in plan["subtasks"]:
        st.markdown(f"- {task}")

    st.markdown("### 🛠️ Suggested Tech Stack")

    st.markdown(", ".join(plan["tech_stack"]))

    selected_task = st.selectbox(
        "Select a subtask to implement:",
        plan["subtasks"]
    )

    if st.button("Generate Code"):

        with st.spinner("⚙️ Generating implementation..."):

            st.session_state.generated_code = generate_code(
                selected_task,
                plan["tech_stack"]
            )

# =========================
# DISPLAY GENERATED CODE
# =========================

if st.session_state.generated_code:

    result = st.session_state.generated_code

    st.markdown(f"## 📄 Generated File: `{result['filename']}`")

    st.code(
        result["code"],
        language=result["filename"].split(".")[-1]
    )

    st.markdown("## 🧠 Explanation")

    st.markdown(result["explanation"])

    if st.button("Review Code"):

        with st.spinner("🔍 Reviewing generated code..."):

            st.session_state.code_review = review_code(
                result["code"]
            )

# =========================
# DISPLAY REVIEW
# =========================

if st.session_state.code_review:

    st.markdown("## 🛠️ Code Review")

    st.markdown(st.session_state.code_review)