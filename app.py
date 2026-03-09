import gradio as gr
from google import genai
from google.genai import types
from PIL import Image
import datetime
import os
import time

# --- 1. SETUP & MODULAR PROMPT LIBRARY ---
API_KEY = os.environ.get('GOOGLE_API_KEY')

try:
    if not API_KEY:
        print("⚠️ Warning: GOOGLE_API_KEY not found. Check your Space Secrets.")
    client = genai.Client(api_key=API_KEY)
except Exception as e:
    print(f"❌ Initialization Error: {e}")

# Stable flash model ID
MODEL_ID = "gemini-2.5-flash"

# --- SYSTEM PROMPT FRAGMENTS ---
BASE_PERSONA = """
ROLE: You are 'Code Mentor,' a Coding Trainer Chatbot intended for use in a high-school programming classroom.
VISION: You are a MULTIMODAL AI. You have vision capabilities. You can seamlessly see, read, and analyze uploaded images, screenshots of code or errors, flowcharts, and architecture diagrams.
Your primary goal is to assist students in learning to program by explaining concepts, guiding problem-solving,
and supporting debugging. You are currently tutoring a student in the '{course}' curriculum, focusing on the '{language}' programming language.
"""

PEDAGOGY_SOCRATIC = """
STRATEGY (SOCRATIC MODE):
- Act like a good instructor, not like Stack Overflow.
- Use scaffolded instruction: hints → partial guidance → full solution (only as an absolute last resort).
- Ask guiding questions to encourage student reasoning and productive struggle before revealing answers.
- Never act as a shortcut solution generator.
"""

PEDAGOGY_DIRECT = """
STRATEGY (DIRECT INSTRUCTION MODE):
- Provide direct, clear explanations of concepts and syntax.
- Use very small code snippets (max 3-5 lines) to demonstrate specific rules.
- Explain the 'WHY' behind the code and how the computer handles it.
- Do not write their entire assignment for them; focus on the specific concept they are stuck on.
"""

CODE_AWARENESS = """
CODE & LANGUAGE CAPABILITIES:
- You fully understand the syntax, semantics, and common beginner mistakes of {language}.
- When evaluating {language} code or reviewing screenshots of code, explain what it does, why it fails, and how to fix it.
- Use simple, precise, age-appropriate explanations, avoiding heavy professional jargon.
"""

ERROR_HANDLING = """
ERROR FOCUS & DEBUGGING-FIRST:
- Treat errors as learning opportunities, not failures.
- Interpret compiler errors, runtime errors, and logic errors in plain English.
- Encourage debugging strategies: code tracing, print statements, test cases, and rubber-duck reasoning.
- Sound like a teacher during a test: "I can help you think through the logic, but I can't write the code for you here."
"""

ADAPTABILITY_AND_TONE = """
ADAPTABILITY & TONE (AFFECTIVE COMPUTING):
- Detect the student's level based on their questions and code complexity, adjusting your vocabulary, pace, and depth.
- Challenge advanced students with "What if..." scenarios, optimization prompts, and edge-case analysis.
- Maintain a patient, non-judgmental, calm, and encouraging tone.
- Use phrases like "You're close" or "This is a common mistake." Never shame or ridicule; normalize confusion.
"""

TRANSPARENCY_AND_ASSESSMENT = """
TRANSPARENCY & ASSESSMENT AWARENESS:
- No Black Boxes: Explain why a solution works. Show step-by-step execution, variable state changes, or call stack evolution.
- Encourage mental models, not memorization.
- Understand AP-style coding task verbs: Predict, Trace, Debug, Modify.
- Can simulate Free-Response Questions, output prediction, and code completion.
- Grade and evaluate the student's *thinking* and logic, not just the correctness of the final code.
- Prevent misuse: Never complete graded assignments for the student. Prioritize student learning over speed of answers.
"""

def build_system_prompt(mode, language, course):
    lang_label = language if language else "General Programming"
    course_label = course if course else "General Computer Science"
    prompt_parts = [BASE_PERSONA.format(course=course_label, language=lang_label)]

    if mode == "Socratic":
        prompt_parts.append(PEDAGOGY_SOCRATIC)
    else:
        prompt_parts.append(PEDAGOGY_DIRECT)

    prompt_parts.append(CODE_AWARENESS.format(language=lang_label))
    prompt_parts.append(ERROR_HANDLING)
    prompt_parts.append(ADAPTABILITY_AND_TONE)
    prompt_parts.append(TRANSPARENCY_AND_ASSESSMENT)
    return "\n\n".join(prompt_parts)

# --- 2. LOGIC FUNCTIONS ---
def chat_logic(message, history, mode, language, course):
    if not language or not course:
        yield "⚠️ **Configuration Required:** Please select a **Course Curriculum** and a **Target Language** from the sidebar before we start coding!"
        return

    current_instruction = build_system_prompt(mode, language, course)
    gemini_history = []
    
    # 1. BUILD HISTORY (Properly passing Images & Text to Memory)
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        raw_content = msg["content"]
        parts_list = []
        
        if isinstance(raw_content, list):
            for item in raw_content:
                if isinstance(item, dict):
                    if item.get("type") == "text" and "text" in item:
                        parts_list.append(types.Part.from_text(text=item["text"]))
                    elif item.get("type") == "file" and "file" in item:
                        path = item["file"].get("path")
                        if path:
                            ext = path.split('.')[-1].lower()
                            if ext in ['png', 'jpg', 'jpeg', 'webp', 'gif']:
                                try:
                                    img = Image.open(path)
                                    parts_list.append(types.Part.from_image(img))
                                except Exception as e:
                                    print(f"Error loading history image: {e}")
                            else:
                                try: # Handle previously uploaded .py, .txt, etc.
                                    with open(path, "r", encoding="utf-8") as f:
                                        parts_list.append(types.Part.from_text(text=f"\n\n--- Uploaded File: {os.path.basename(path)} ---\n{f.read()}"))
                                except:
                                    pass
        else:
            text_content = str(raw_content)
            if text_content.strip():
                parts_list.append(types.Part.from_text(text=text_content))
                
        if parts_list:
            gemini_history.append(types.Content(role=role, parts=parts_list))

    try:
        chat = client.chats.create(
            model=MODEL_ID,
            config=types.GenerateContentConfig(
                system_instruction=current_instruction,
                temperature=0.7 if mode == "Socratic" else 0.2
            ),
            history=gemini_history
        )

        # 2. CURRENT MESSAGE PAYLOAD
        user_text = message.get("text", "")
        user_files = message.get("files", [])
        
        payload = []
        if user_text.strip():
            payload.append(user_text)
            
        for file_item in user_files:
            path = file_item.get("path") if isinstance(file_item, dict) else file_item
            ext = path.split('.')[-1].lower()
            
            if ext in ['png', 'jpg', 'jpeg', 'webp', 'gif']:
                try:
                    img = Image.open(path)
                    payload.append(img)
                except Exception as e:
                    print(f"Error loading image: {e}")
            else:
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        file_text = f.read()
                        payload.append(f"\n\n--- Uploaded File: {os.path.basename(path)} ---\n{file_text}")
                except Exception as ex:
                    print(f"Could not read file {path}: {ex}")

        if not payload:
            yield "⚠️ Please provide some text or an image."
            return

        response_stream = chat.send_message_stream(payload)
        full_response = ""
        for chunk in response_stream:
            if chunk.text:
                for char in chunk.text:
                    full_response += char
                    yield full_response
                    time.sleep(0.015)
    except Exception as e:
        yield f"🤖 Technical Hiccup: {str(e)}"

def save_transcript(history):
    if not history: return None
    filename = f"DACodeX_Transcript_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt"
    transcript_text = "DACODEX MENTOR SESSION\n" + "="*30 + "\n\n"
    
    for msg in history:
        prefix = "STUDENT" if msg["role"] == "user" else "MENTOR"
        raw_content = msg["content"]
        
        if isinstance(raw_content, list):
            texts = []
            for item in raw_content:
                if isinstance(item, dict):
                    if item.get("type") == "text" and "text" in item:
                        texts.append(item["text"])
                    elif item.get("type") == "file" and "file" in item:
                        path = item["file"].get("path")
                        if path:
                            ext = path.split('.')[-1].lower()
                            if ext in ['png', 'jpg', 'jpeg', 'webp', 'gif']:
                                texts.append(f"[Uploaded Image: {os.path.basename(path)}]")
                            else:
                                texts.append(f"[Uploaded File: {os.path.basename(path)}]")
            text_content = "\n".join(texts)
        else:
            text_content = str(raw_content)
            
        transcript_text += f"{prefix}:\n{text_content}\n\n"
        
    with open(filename, "w", encoding="utf-8") as f:
        f.write(transcript_text)
    return filename

def archive_and_clear(history, current_storage):
    if not history: return current_storage, [], gr.update(choices=[item[0] for item in current_storage])
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    label = f"Session {timestamp} ({len(history)} messages)"
    new_storage = [(label, history)] + current_storage
    new_choices = [item[0] for item in new_storage]
    return new_storage, [], gr.update(choices=new_choices, value=None)

def load_from_history(selected_label, current_storage):
    if not selected_label: return gr.update()
    for label, history in current_storage:
        if label == selected_label: return history
    return []

def toggle_sidebar_func(is_visible):
    new_state = not is_visible
    button_text = "◀ Hide Sidebar" if new_state else "▶ Show Sidebar"
    return new_state, gr.update(visible=new_state), gr.update(value=button_text)

# --- 3. THEME & ADVANCED CSS ---
dacodex_theme = gr.themes.Base(
    primary_hue=gr.themes.colors.red,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill="#09090b",
    block_background_fill="#121217",
    block_border_color="#27272a",
    body_text_color="#e4e4e7",
    button_primary_background_fill="#dc2626",
    button_primary_background_fill_hover="#ef4444",
    block_label_text_color="#D1D5DB"
)

custom_css = """
@keyframes flicker {
    0% { opacity: 0.97; }
    5% { opacity: 0.9; }
    10% { opacity: 0.97; }
    100% { opacity: 1; }
}
.landing-container {
    height: 90vh;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    background: radial-gradient(circle at center, #1e1b4b 0%, #09090b 100%);
    animation: flicker 0.15s infinite;
    text-align: center;
}
.start-btn {
    border: 1px solid #ef4444 !important;
    box-shadow: 0 0 15px rgba(220, 38, 38, 0.4);
    letter-spacing: 2px;
    font-weight: bold !important;
    padding: 20px 40px !important;
    font-size: 1.2em !important;
    transition: all 0.3s ease !important;
    margin-top: 20px;
    cursor: pointer;
}
.start-btn:hover {
    box-shadow: 0 0 30px rgba(239, 68, 68, 0.8);
    transform: scale(1.05) !important;
}
#chatbot-window {
    border-left: 2px solid #dc2626;
    background: rgba(18, 18, 23, 0.8);
}
.info-popup {
    background: #1a1a23;
    border: 1px solid #dc2626;
    padding: 15px;
    border-radius: 8px;
    margin-bottom: 15px;
    box-shadow: 0 0 10px rgba(220, 38, 38, 0.2);
}
.sidebar-btn {
    margin-top: 10px !important;
}
.toggle-btn {
    width: 150px !important;
    margin-bottom: 10px !important;
}
"""

def get_logo(width=400, height=100):
    return f"""
    <div style="display: flex; justify-content: center; align-items: center; padding: 20px 0;">
        <svg width="{width}" height="{height}" viewBox="0 0 400 100" fill="none" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <filter id="neonRed" x="-20%" y="-20%" width="140%" height="140%">
                    <feGaussianBlur stdDeviation="3" result="blur" />
                    <feComposite in="SourceGraphic" in2="blur" operator="over" />
                </filter>
            </defs>
            <path d="M40 30L20 50L40 70" stroke="#dc2626" stroke-width="5" stroke-linecap="round" filter="url(#neonRed)"/>
            <path d="M70 30L90 50L70 70" stroke="#dc2626" stroke-width="5" stroke-linecap="round" filter="url(#neonRed)"/>
            <text x="100" y="65" fill="white" style="font-family:'JetBrains Mono', monospace; font-weight:800; font-size:45px;">DA</text>
            <text x="165" y="65" fill="#dc2626" style="font-family:'JetBrains Mono', monospace; font-weight:800; font-size:45px;" filter="url(#neonRed)">CODE</text>
            <text x="285" y="65" fill="white" style="font-family:'JetBrains Mono', monospace; font-weight:200; font-size:45px;">X</text>
            <rect x="100" y="75" width="230" height="2" fill="#dc2626" fill-opacity="0.3"/>
        </svg>
    </div>
    """

# --- 4. BUILD UI ---
with gr.Blocks() as demo:
    session_storage = gr.State([])
    sidebar_state = gr.State(True)

    # PRE-SCREEN (Landing Page)
    with gr.Column(visible=True, elem_classes="landing-container") as landing_page:
        gr.HTML(get_logo(width=600, height=150))
        gr.Markdown("### // SYSTEM STATUS: ONLINE\n// ACADEMIC CORE: READY")
        start_button = gr.Button("INITIALIZE INTERFACE", variant="primary", elem_classes="start-btn")

    # MAIN APP (The Chat Interface)
    with gr.Column(visible=False) as main_app:
        gr.HTML(get_logo(width=300, height=80))

        with gr.Row():
            with gr.Column(scale=1) as sidebar_col:
                # Info Popup Section
                info_btn = gr.Button("ℹ️ > Quick Guide", size="sm", variant="secondary")

                with gr.Column(visible=False, elem_classes="info-popup") as info_panel:
                    gr.Markdown("""
                    **<u>Teaching Protocol:</u>**
                    * **Socratic:** AI hints and asks questions to guide you.
                    * **Direct:** AI explains concepts and gives examples immediately.
                    **<u>Upload Images & Code:</u>**
                    Use the 📎 icon in the chat bar to upload screenshots of errors, flowcharts, or even raw `.py` files!
                    **<u>Archive Current Session:</u>**
                    Saves current chat in 'Previous Chats' and creates a new session.
                    """)
                    close_info_btn = gr.Button("Close Guide", size="sm")

                gr.Markdown("---")
                mode_selector = gr.Radio(choices=["Socratic", "Direct"], value="Socratic", label="Teaching Protocol")
                course_selector = gr.Dropdown(choices=["AP CS A", "AP CSP", "C++ Fundamentals", "Web Development 101", "Intro to Python", "AP Cybersecurity", "Other"], value="Intro to Python", label="Course Curriculum")
                language_selector = gr.Dropdown(choices=["Java", "Python", "JavaScript", "C++", "C#", "SQL"], value="Python", label="Target Language")

                gr.Markdown("---")
                gr.Markdown("### Session Archives")
                history_dropdown = gr.Dropdown(choices=[], label="Previous Chats", interactive=True)
                clear_btn = gr.Button("Archive Current Session", variant="secondary", elem_classes="sidebar-btn")

                gr.Markdown("---")
                download_btn = gr.Button("Download Text File", variant="primary", elem_classes="sidebar-btn")
                transcript_file = gr.File(label="Download Ready", visible=False)

            with gr.Column(scale=4):
                toggle_sidebar_btn = gr.Button("◀ Hide Sidebar", size="sm", elem_classes="toggle-btn")
                
                chat_ui = gr.ChatInterface(
                    fn=chat_logic,
                    additional_inputs=[mode_selector, language_selector, course_selector],
                    chatbot=gr.Chatbot(height=600, elem_id="chatbot-window", label="DACodeX"),
                    multimodal=True
                )

    # --- UI LOGIC / EVENTS ---
    def start_app():
        return gr.update(visible=False), gr.update(visible=True)

    def toggle_info(show):
        return gr.update(visible=show)

    start_button.click(fn=start_app, outputs=[landing_page, main_app])

    info_btn.click(fn=lambda: toggle_info(True), outputs=info_panel)
    close_info_btn.click(fn=lambda: toggle_info(False), outputs=info_panel)

    clear_btn.click(
        archive_and_clear,
        inputs=[chat_ui.chatbot, session_storage],
        outputs=[session_storage, chat_ui.chatbot, history_dropdown]
    )

    download_btn.click(
        save_transcript,
        inputs=[chat_ui.chatbot],
        outputs=[transcript_file]
    ).then(
        lambda: gr.update(visible=True), None, transcript_file
    )

    history_dropdown.change(
        load_from_history,
        inputs=[history_dropdown, session_storage],
        outputs=[chat_ui.chatbot]
    )

    toggle_sidebar_btn.click(
        toggle_sidebar_func,
        inputs=[sidebar_state],
        outputs=[sidebar_state, sidebar_col, toggle_sidebar_btn]
    )

if __name__ == "__main__":
    demo.launch(theme=dacodex_theme, css=custom_css)
