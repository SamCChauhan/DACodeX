from nicegui import ui, app, events
from google import genai
from google.genai import types
from PIL import Image
import datetime
import os
import asyncio
import io

# --- 1. SETUP & MODULAR PROMPT LIBRARY ---
API_KEY = os.environ.get('GOOGLE_API_KEY')

try:
    if not API_KEY:
        print("⚠️ Warning: GOOGLE_API_KEY not found. Set it in your environment variables.")
    # Using the async client for non-blocking UI updates in NiceGUI
    client = genai.Client(api_key=API_KEY)
except Exception as e:
    print(f"❌ Initialization Error: {e}")

MODEL_ID = "gemini-2.5-flash"

# --- SYSTEM PROMPT FRAGMENTS (PRESERVED) ---
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


# --- STATE MANAGEMENT ---
chat_history = []  # Stores UI messages
session_storage = {} # Stores archived sessions
pending_uploads = [] # Temporary storage for files before sending

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

# --- UI COMPONENTS & LOGIC ---

@ui.page('/')
def main_page():
    # --- STYLING ---
    ui.add_css("""
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;800&display=swap');
        body { background-color: #09090b; color: #e4e4e7; font-family: 'JetBrains Mono', monospace; }
        
        @keyframes flicker {
            0% { opacity: 0.97; } 5% { opacity: 0.9; } 10% { opacity: 0.97; } 100% { opacity: 1; }
        }
        .landing-container {
            height: 100vh;
            background: radial-gradient(circle at center, #1e1b4b 0%, #09090b 100%);
            animation: flicker 0.15s infinite;
        }
        .start-btn {
            border: 1px solid #ef4444 !important;
            box-shadow: 0 0 15px rgba(220, 38, 38, 0.4);
            letter-spacing: 2px;
            transition: all 0.3s ease !important;
        }
        .start-btn:hover {
            box-shadow: 0 0 30px rgba(239, 68, 68, 0.8);
            transform: scale(1.05) !important;
        }
        
        /* Message Text Colors */
        .q-message-text { background-color: #121217 !important; border: 1px solid #27272a; }
        .q-message-text--sent { background-color: #dc2626 !important; border: none; }
        .q-message-name { color: #D1D5DB !important; }
        
        /* === MARKDOWN SPECIFIC STYLING === */
        .q-message-text-content { color: #ffffff !important; }
        .q-message-text-content p { margin: 0 0 0.5em 0; color: #ffffff !important; }
        .q-message-text-content p:last-child { margin-bottom: 0; }
        .q-message-text-content a { color: #ef4444; font-weight: bold; }
        
        /* Lists Fix for Quasar Reset */
        .q-message-text-content ul {
            list-style-type: disc !important;
            padding-left: 1.5em !important;
            margin-top: 0.5em !important;
            margin-bottom: 0.5em !important;
        }
        .q-message-text-content ol {
            list-style-type: decimal !important;
            padding-left: 1.5em !important;
            margin-top: 0.5em !important;
            margin-bottom: 0.5em !important;
        }
        .q-message-text-content li {
            display: list-item !important;
            margin-bottom: 0.25em !important;
            color: #ffffff !important;
        }
        
        /* Inline code (e.g., `print()`) */
        .q-message-text-content :not(pre) > code { 
            background-color: #27272a; 
            color: #ffb3c1; 
            padding: 2px 6px; 
            border-radius: 4px; 
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9em;
        }
        
        /* Code blocks (e.g., ```python ... ```) */
        .q-message-text-content pre { 
            background-color: #09090b !important; 
            border: 1px solid #27272a; 
            padding: 12px; 
            border-radius: 8px; 
            overflow-x: auto;
            margin: 0.5em 0;
        }
        .q-message-text-content pre code { 
            color: #e4e4e7; 
            background-color: transparent; 
            padding: 0; 
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9em;
        }
        /* ================================= */
        
        .drawer-bg { background-color: #121217 !important; border-right: 1px solid #27272a; }
    """)

    ui.colors(primary='#dc2626', secondary='#121217', accent='#ef4444')

    # --- 1. LANDING PAGE ---
    with ui.column().classes('w-full items-center justify-center landing-container') as landing_view:
        ui.html(get_logo(width=600, height=150))
        ui.markdown("### // SYSTEM STATUS: ONLINE\n// ACADEMIC CORE: READY").classes('text-center')
        start_btn = ui.button("INITIALIZE INTERFACE").classes('start-btn mt-4 px-8 py-4 text-lg font-bold rounded text-white')

    # --- 2. SIDEBAR ---
    with ui.left_drawer(value=False).classes('drawer-bg p-4') as drawer:
        ui.html(get_logo(width=200, height=60)).classes('mb-4')
        
        with ui.dialog() as info_dialog, ui.card().classes('bg-[#1a1a23] border border-[#dc2626] text-white'):
            ui.markdown("""
            **<u>Teaching Protocol:</u>**
            * **Socratic:** AI hints and asks questions to guide you.
            * **Direct:** AI explains concepts and gives examples immediately.
            **<u>Upload Images & Code:</u>**
            Use the 📎 icon in the chat bar to upload screenshots of errors, flowcharts, or even raw `.py` files!
            **<u>Archive Current Session:</u>**
            Saves current chat in 'Previous Chats' and creates a new session.
            """)
            ui.button('Close', on_click=info_dialog.close)
        
        ui.button("ℹ️ Quick Guide", on_click=info_dialog.open).props('outline rounded size=sm').classes('w-full mb-4 text-white')
        ui.separator()
        
        mode_select = ui.select(["Socratic", "Direct"], value="Socratic", label="Teaching Protocol").classes('w-full mt-2 text-white')
        course_select = ui.select(["AP CS A", "AP CSP", "C++ Fundamentals", "Web Development 101", "Intro to Python", "AP Cybersecurity", "Other"], value="Intro to Python", label="Course Curriculum").classes('w-full mt-2 text-white')
        language_select = ui.select(["Java", "Python", "JavaScript", "C++", "C#", "SQL"], value="Python", label="Target Language").classes('w-full mt-2 text-white')
        
        ui.separator().classes('my-4')
        ui.label("Session Archives").classes('text-lg font-bold text-gray-300')
        
        history_dropdown = ui.select([], label="Previous Chats").classes('w-full mt-2 text-white')
        
        def archive_session():
            if not chat_history: return
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            label = f"Session {timestamp} ({len(chat_history)} msgs)"
            session_storage[label] = chat_history.copy()
            history_dropdown.options = list(session_storage.keys())
            history_dropdown.update()
            chat_history.clear()
            render_messages.refresh()
        
        ui.button("Archive Current Session", on_click=archive_session).props('outline rounded').classes('w-full mt-2 text-white')
        
        def load_session(e):
            if e.value in session_storage:
                chat_history.clear()
                chat_history.extend(session_storage[e.value])
                render_messages.refresh()

        history_dropdown.on_value_change(load_session)
        
        ui.separator().classes('my-4')
        
        def download_transcript():
            if not chat_history: return
            transcript_text = "DACODEX MENTOR SESSION\n" + "="*30 + "\n\n"
            for msg in chat_history:
                prefix = "STUDENT" if msg["role"] == "user" else "MENTOR"
                transcript_text += f"{prefix}:\n{msg['raw_text']}\n\n"
            
            # Create file bytes for download
            filename = f"DACodeX_Transcript_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt"
            file_bytes = transcript_text.encode('utf-8')
            ui.download(file_bytes, filename)
            
        ui.button("Download Text File", on_click=download_transcript).classes('w-full mt-2 start-btn text-white')

    # --- 3. MAIN CHAT AREA ---
    with ui.column().classes('w-full h-screen relative') as main_chat_view:
        main_chat_view.set_visibility(False)
        
        # Header row for drawer toggle
        with ui.row().classes('w-full p-4 border-b border-[#27272a] bg-[#121217] items-center z-10'):
            ui.button(icon='menu', on_click=drawer.toggle).props('flat round dense color=white')
            ui.label('DACodeX - Academic Core').classes('text-xl font-bold ml-2 text-white')

        # Chat Messages Area
        with ui.scroll_area().classes('flex-grow w-full p-4 pb-32') as scroll_area:
            @ui.refreshable
            def render_messages():
                for msg in chat_history:
                    # Note: We NO LONGER pass text=msg['text'] here. 
                    # We pass the text into ui.markdown() inside the context manager instead!
                    with ui.chat_message(name=msg['name'], sent=msg['sent']):
                        # Renders the text as rich Markdown with support for breaks and cuddled lists
                        ui.markdown(msg['text'], extras=['fenced-code-blocks', 'tables', 'cuddled-lists', 'breaks'])
                        
                        # Render images if attached
                        for img_html in msg.get('images', []):
                            ui.html(img_html).classes('max-w-xs rounded mt-2')
            
            render_messages()

        # Input Area (Pinned to bottom)
        with ui.row().classes('absolute bottom-0 w-full p-4 bg-[#09090b] border-t border-[#27272a] items-end z-10'):
            
            def handle_upload(e: events.UploadEventArguments):
                filename = e.name
                ext = filename.split('.')[-1].lower()
                content = e.content.read()
                
                if ext in ['png', 'jpg', 'jpeg', 'webp', 'gif']:
                    try:
                        img = Image.open(io.BytesIO(content))
                        pending_uploads.append({'type': 'image', 'data': img, 'name': filename})
                        ui.notify(f"Attached Image: {filename}", type='positive')
                    except Exception as ex:
                        ui.notify(f"Error loading image: {ex}", color='negative')
                else:
                    try:
                        text_content = content.decode('utf-8', errors='ignore')
                        pending_uploads.append({'type': 'text', 'data': f"\n\n--- Uploaded File: {filename} ---\n{text_content}", 'name': filename})
                        ui.notify(f"Attached File: {filename}", type='positive')
                    except Exception as ex:
                        ui.notify(f"Could not read file {filename}: {ex}", color='negative')
                
                upload_element.reset()

            # The invisible uploader component
            upload_element = ui.upload(multiple=True, auto_upload=True, on_upload=handle_upload).classes('absolute w-0 h-0 opacity-0 overflow-hidden -z-10')
            
            # The visible icon button that triggers the hidden uploader's file dialog via JS
            ui.button(icon='attach_file', on_click=lambda: ui.run_javascript('document.querySelector(".q-uploader__input")?.click()')).props('flat round dense color=white').classes('mb-2')

            text_input = ui.input(placeholder="Type your message...").classes('flex-grow mx-2').props('outlined dark rounded')
            
            async def send_message():
                user_text = text_input.value.strip()
                if not user_text and not pending_uploads:
                    ui.notify("Please provide some text or an image.", color='warning')
                    return
                    
                # 1. Build Payload and UI Message
                payload = []
                images_for_ui = []
                raw_text_record = user_text
                
                if user_text:
                    payload.append(user_text)
                    
                for item in pending_uploads:
                    if item['type'] == 'image':
                        payload.append(item['data'])
                        raw_text_record += f"\n[Uploaded Image: {item['name']}]"
                        # For UI display, convert PIL to base64
                        import base64
                        buffered = io.BytesIO()
                        item['data'].save(buffered, format="PNG")
                        img_str = base64.b64encode(buffered.getvalue()).decode()
                        images_for_ui.append(f'<img src="data:image/png;base64,{img_str}" />')
                    elif item['type'] == 'text':
                        payload.append(item['data'])
                        raw_text_record += f"\n[Uploaded File: {item['name']}]"

                # 2. Add User Message to UI
                chat_history.append({
                    'text': user_text if user_text else "📎 *(Attachments sent)*", 
                    'name': 'Student', 
                    'sent': True, 
                    'role': 'user',
                    'raw_text': raw_text_record,
                    'images': images_for_ui
                })
                
                text_input.value = ""
                pending_uploads.clear()
                render_messages.refresh()
                scroll_area.scroll_to(percent=1)

                # 3. Setup GenAI API Call Structure
                current_instruction = build_system_prompt(mode_select.value, language_select.value, course_select.value)
                
                gemini_history = []
                for msg in chat_history[:-1]:  # Exclude the message we just added (it goes in payload)
                    role = msg['role']
                    # Simplified history recreation for API
                    gemini_history.append(types.Content(role=role, parts=[types.Part.from_text(text=msg['raw_text'])]))

                try:
                    # 4. Async API Call to prevent GUI Freezing
                    chat = client.aio.chats.create(
                        model=MODEL_ID,
                        config=types.GenerateContentConfig(
                            system_instruction=current_instruction,
                            temperature=0.7 if mode_select.value == "Socratic" else 0.2
                        ),
                        history=gemini_history
                    )
                    
                    # Add empty AI message to UI
                    chat_history.append({'text': '', 'name': 'DACodeX', 'sent': False, 'role': 'model', 'raw_text': ''})
                    render_messages.refresh()
                    scroll_area.scroll_to(percent=1)
                    
                    response_stream = await chat.send_message_stream(payload)
                    full_response = ""
                    
                    async for chunk in response_stream:
                        if chunk.text:
                            full_response += chunk.text
                            chat_history[-1]['text'] = full_response
                            chat_history[-1]['raw_text'] = full_response
                            render_messages.refresh()
                            scroll_area.scroll_to(percent=1)
                            
                except Exception as e:
                    ui.notify(f"🤖 Technical Hiccup: {str(e)}", color='negative')
            
            text_input.on('keydown.enter', send_message)
            ui.button(icon='send', on_click=send_message).props('flat round dense color=primary').classes('mb-2')

    # --- 4. INTERFACE STARTUP LOGIC ---
    def start_interface():
        landing_view.set_visibility(False)
        main_chat_view.set_visibility(True)
        drawer.value = True # Triggers the sidebar to slide out smoothly
        
    # Wire the button to the function we just defined
    start_btn.on_click(start_interface)


# --- INITIALIZATION (NATIVE DESKTOP MODE) ---
if __name__ in {"__main__", "__mp_main__"}:
    # Runs the application as a standalone desktop window using PyWebView
    ui.run(
        native=True, 
        window_size=(1200, 800), 
        title="DACodeX - Academic Core", 
        dark=True,
        show=True
    )
