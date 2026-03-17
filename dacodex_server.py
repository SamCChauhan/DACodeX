from nicegui import ui, app, events
from google import genai
from google.genai import types
from PIL import Image
import datetime
import os
import asyncio
import io
import base64

# --- 1. SETUP & MODULAR PROMPT LIBRARY ---
API_KEY = os.environ.get('GOOGLE_API_KEY')
try:
    if not API_KEY:
        print("⚠️ Warning: GOOGLE_API_KEY not found. Set it in your environment variables.")
    # Using the async client for non-blocking UI updates in NiceGUI
    client = genai.Client(api_key=API_KEY)
except Exception as e:
    print(f"❌ Initialization Error: {e}")

MODEL_ID = "gemini-2.5-flash-lite"

# --- SYSTEM PROMPT FRAGMENTS (PRESERVED) ---
BASE_PERSONA = """ROLE: You are 'Code Mentor,' a Coding Trainer Chatbot intended for use in a high-school programming classroom.
VISION: You are a MULTIMODAL AI. You have vision capabilities. You can seamlessly see, read, and analyze uploaded images, screenshots of code or errors, flowcharts, and architecture diagrams.
Your primary goal is to assist students in learning to program by explaining concepts, guiding problem-solving,
and supporting debugging. You are currently tutoring a student in the '{course}' curriculum, focusing on the '{language}' programming language."""

PEDAGOGY_SOCRATIC = """STRATEGY (SOCRATIC MODE):
- Act like a good instructor, not like Stack Overflow.
- Use scaffolded instruction: hints → partial guidance → full solution (only as an absolute last resort).
- Ask guiding questions to encourage student reasoning and productive struggle before revealing answers.
- Never act as a shortcut solution generator."""

PEDAGOGY_DIRECT = """STRATEGY (DIRECT INSTRUCTION MODE):
- Provide direct, clear explanations of concepts and syntax.
- Use very small code snippets (max 3-5 lines) to demonstrate specific rules.
- Explain the 'WHY' behind the code and how the computer handles it.
- Do not write their entire assignment for them; focus on the specific concept they are stuck on."""

CODE_AWARENESS = """CODE & LANGUAGE CAPABILITIES:
- You fully understand the syntax, semantics, and common beginner mistakes of {language}.
- When evaluating {language} code or reviewing screenshots of code, explain what it does, why it fails, and how to fix it.
- Use simple, precise, age-appropriate explanations, avoiding heavy professional jargon."""

ERROR_HANDLING = """ERROR FOCUS & DEBUGGING-FIRST:
- Treat errors as learning opportunities, not failures.
- Interpret compiler errors, runtime errors, and logic errors in plain English.
- Encourage debugging strategies: code tracing, print statements, test cases, and rubber-duck reasoning.
- Sound like a teacher during a test: "I can help you think through the logic, but I can't write the code for you here." """

ADAPTABILITY_AND_TONE = """ADAPTABILITY & TONE (AFFECTIVE COMPUTING):
- Detect the student's level based on their questions and code complexity, adjusting your vocabulary, pace, and depth.
- Challenge advanced students with "What if..." scenarios, optimization prompts, and edge-case analysis.
- Maintain a patient, non-judgmental, calm, and encouraging tone.
- Use phrases like "You're close" or "This is a common mistake." Never shame or ridicule; normalize confusion."""

TRANSPARENCY_AND_ASSESSMENT = """TRANSPARENCY & ASSESSMENT AWARENESS:
- No Black Boxes: Explain why a solution works. Show step-by-step execution, variable state changes, or call stack evolution.
- Encourage mental models, not memorization.
- Understand AP-style coding task verbs: Predict, Trace, Debug, Modify.
- Can simulate Free-Response Questions, output prediction, and code completion.
- Grade and evaluate the student's *thinking* and logic, not just the correctness of the final code.
- Prevent misuse: Never complete graded assignments for the student. Prioritize student learning over speed of answers."""

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

def get_logo(width=400, height=100):
    return f"""
    <div style="display: flex; justify-content: center; align-items: center; padding: 20px 0;">
        <svg width="{width}" height="{height}" viewBox="0 0 400 100" fill="none" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <filter id="neonRed" x="-20%" y="-20%" width="140%" height="140%">
                    <feGaussianBlur stdDeviation="3" result="blur" />
                    <feDropShadow dx="0" dy="0" stdDeviation="5" flood-color="#dc2626" />
                    <feComposite in="SourceGraphic" in2="blur" operator="over" />
                </filter>
            </defs>
            <path d="M40 30L20 50L40 70" stroke="#dc2626" stroke-width="5" stroke-linecap="round" filter="url(#neonRed)"/>
            <path d="M70 30L90 50L70 70" stroke="#dc2626" stroke-width="5" stroke-linecap="round" filter="url(#neonRed)"/>
            <text x="100" y="65" fill="#ffffff" style="font-family:'JetBrains Mono', monospace; font-weight:800; font-size:45px;">DA</text>
            <text x="165" y="65" fill="#dc2626" style="font-family:'JetBrains Mono', monospace; font-weight:800; font-size:45px;" filter="url(#neonRed)">CODE</text>
            <text x="285" y="65" fill="#ffffff" style="font-family:'JetBrains Mono', monospace; font-weight:200; font-size:45px;">X</text>
            <rect x="100" y="75" width="230" height="2" fill="#dc2626" fill-opacity="0.5"/>
        </svg>
    </div>
    """

# --- UI COMPONENTS & LOGIC ---
@ui.page('/')
def main_page():
    # WEB MODIFICATION: State lists moved inside the page function!
    # This ensures that when User A and User B connect from different computers,
    # they get their own isolated variables and don't share chats.
    chat_history = []  
    session_storage = {} 
    pending_uploads = [] 

    # --- STYLING (RED THEME RESTORED) ---
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
            border: 1px solid #dc2626 !important;
            box-shadow: 0 0 15px rgba(220, 38, 38, 0.4);
            letter-spacing: 2px;
            transition: all 0.3s ease !important;
        }
        .start-btn:hover {
            box-shadow: 0 0 30px rgba(220, 38, 38, 0.8);
            transform: scale(1.05) !important;
        }
        
        /* Message Text Colors (RED) */
        .q-message-text { background-color: #121217 !important; border: 1px solid #27272a; position: relative; }
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
             position: relative;
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

        /* Copy Button Style */
        .copy-btn {
            position: absolute;
            top: 5px;
            right: 5px;
            padding: 4px 8px;
            background: #27272a;
            color: #e4e4e7;
            border: 1px solid #dc2626;
            border-radius: 4px;
            font-size: 10px;
            cursor: pointer;
            z-index: 10;
            opacity: 0.6;
            transition: opacity 0.2s;
        }
        .copy-btn:hover { opacity: 1; background: #dc2626; }
        /* ================================= */
        
        .drawer-bg { background-color: #121217 !important; border-left: 1px solid #27272a; }
    """)
    ui.colors(primary='#dc2626', secondary='#121217', accent='#ef4444')

    ui.add_head_html("""
        <script>
        function copyCode(btn) {
            const pre = btn.parentElement;
            const code = pre.querySelector('code').innerText;
            navigator.clipboard.writeText(code).then(() => {
                const oldText = btn.innerText;
                btn.innerText = 'COPIED!';
                setTimeout(() => { btn.innerText = oldText; }, 2000);
            });
        }

        const observer = new MutationObserver((mutations) => {
            document.querySelectorAll('pre:not(.has-copy-btn)').forEach((pre) => {
                pre.classList.add('has-copy-btn');
                const btn = document.createElement('button');
                btn.className = 'copy-btn';
                btn.innerText = 'COPY';
                btn.onclick = function() { copyCode(this); };
                pre.appendChild(btn);
            });
        });

        document.addEventListener('DOMContentLoaded', () => {
            observer.observe(document.body, { childList: true, subtree: true });
        });
        </script>
    """)

    # --- 1. LANDING PAGE ---
    with ui.column().classes('w-full items-center justify-center landing-container') as landing_view:
        ui.html(get_logo(width=600, height=150))
        ui.markdown("### // SYSTEM STATUS: ONLINE\n// ACADEMIC CORE: READY").classes('text-center')
        start_btn = ui.button("INITIALIZE INTERFACE").classes('start-btn mt-4 px-8 py-4 text-lg font-bold rounded text-white')

    # --- 2. SIDEBAR ---
    with ui.right_drawer(value=False).classes('drawer-bg p-4') as drawer:
        ui.html(get_logo(width=200, height=60)).classes('mb-4')
        
        with ui.dialog() as info_dialog, ui.card().classes('bg-[#1a1a23] border border-[#dc2626] text-white'):
            ui.markdown("""
**<u>Teaching Protocol:</u>**

* **Socratic:** AI hints and asks questions to guide you.
* **Direct:** AI explains concepts and gives examples immediately.

**<u>Upload Images & Code:</u>**

* Use the 📎 icon in the chat bar to upload screenshots of errors, flowcharts, or even raw `.py` files!

**<u>Archive Current Session:</u>**

* Saves current chat in 'Previous Chats' and creates a new session.
            """).classes('p-4')
            ui.button('Close', on_click=info_dialog.close).classes('mt-2')
        
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
        
        # WEB MODIFICATION: File downloads via browser instead of OS file paths
        def download_transcript():
            if not chat_history:
                ui.notify("No chat history to save.", type="warning")
                return
            
            transcript_text = "DACODEX MENTOR SESSION\n" + "="*30 + "\n\n"
            for msg in chat_history:
                prefix = "STUDENT" if msg["role"] == "user" else "MENTOR"
                transcript_text += f"{prefix}:\n{msg['raw_text']}\n\n"
            
            filename = f"DACodeX_Transcript_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt"
            
            # Encodes text into memory and triggers a prompt in the User's Web Browser
            ui.download(transcript_text.encode('utf-8'), filename)
            ui.notify("Download initiated!", type='positive')
            
        ui.button("Download Text File", on_click=download_transcript).classes('w-full mt-2 start-btn text-white')

    # --- 3. MAIN CHAT AREA ---
    with ui.column().classes('w-full h-screen relative') as main_chat_view:
        main_chat_view.set_visibility(False)
        
        with ui.row().classes('w-full p-4 border-b border-[#27272a] bg-[#121217] items-center justify-between z-10'):
            ui.label('DACodeX - Coding Assistant').classes('text-xl font-bold ml-2 text-white')
            ui.button(icon='menu', on_click=drawer.toggle).props('flat round dense color=white')

        with ui.scroll_area().classes('flex-grow w-full p-4 pb-40') as scroll_area:
            @ui.refreshable
            def render_messages():
                for index, msg in enumerate(chat_history):
                    with ui.chat_message(name=msg['name'], sent=msg['sent']):
                        ui.markdown(msg['text'], extras=['fenced-code-blocks', 'tables', 'cuddled-lists', 'breaks'])
                        for img_html in msg.get('images', []):
                            ui.html(img_html).classes('max-w-xs rounded mt-2')

            render_messages()

        # WEB MODIFICATION: Replaced native OS file picker with a Web Upload Component
        with ui.dialog() as upload_dialog, ui.card().classes('bg-[#121217] border border-[#27272a] text-white'):
            ui.label('Upload Reference Material').classes('text-lg font-bold')
            
            def handle_web_upload(e: events.UploadEventArguments):
                try:
                    content_bytes = e.content.read()
                    filename = e.name
                    ext = filename.split('.')[-1].lower() if '.' in filename else ''
                    
                    if ext in ['png', 'jpg', 'jpeg', 'webp', 'gif']:
                        img = Image.open(io.BytesIO(content_bytes))
                        pending_uploads.append({'type': 'image', 'data': img, 'name': filename})
                        ui.notify(f"Attached Image: {filename}", type='positive')
                    else:
                        text_content = content_bytes.decode('utf-8', errors='ignore')
                        pending_uploads.append({'type': 'text', 'data': f"\n\n--- Uploaded File: {filename} ---\n{text_content}", 'name': filename})
                        ui.notify(f"Attached File: {filename}", type='positive')
                    
                    render_previews.refresh()
                    upload_dialog.close()
                except Exception as ex:
                    ui.notify(f"Upload failed: {str(ex)}", color='negative')

            ui.upload(multiple=True, on_upload=handle_web_upload, auto_upload=True).classes('max-w-full')
            ui.button('Close', on_click=upload_dialog.close).props('outline rounded').classes('w-full mt-2')


        # --- 4. UNIFIED CHAT INPUT BOX ---
        with ui.column().classes('absolute bottom-0 w-full p-4 bg-[#09090b] border-t border-[#27272a] z-10'):
            
            # Unified Container Box
            with ui.column().classes('w-full bg-[#121217] border border-[#27272a] rounded-xl p-1 gap-0'):
                
                # Dynamic Preview Area
                @ui.refreshable
                def render_previews():
                    if pending_uploads:
                        with ui.row().classes('w-full gap-3 px-3 pt-3 pb-1 overflow-x-auto no-wrap'):
                            for idx, item in enumerate(pending_uploads):
                                with ui.card().classes('w-16 h-16 p-0 bg-[#09090b] border border-[#3f3f46] rounded-lg relative shadow-none flex-shrink-0 flex items-center justify-center'):
                                    if item['type'] == 'image':
                                        buffered = io.BytesIO()
                                        item['data'].save(buffered, format="PNG")
                                        img_str = base64.b64encode(buffered.getvalue()).decode()
                                        ui.html(f'<img src="data:image/png;base64,{img_str}" style="width: 100%; height: 100%; object-fit: cover; border-radius: 6px;" />')
                                    else:
                                        ui.label('📄').classes('text-2xl')
                                    
                                    def remove_item(i=idx):
                                        pending_uploads.pop(i)
                                        render_previews.refresh()
                                    
                                    ui.button(icon='close', on_click=remove_item).props('flat round dense size=xs color=white').classes('absolute -top-2 -right-2 bg-[#dc2626] rounded-full z-10 w-5 h-5 min-h-0 min-w-0 p-0 shadow')
                
                render_previews()

                # Text Input Row
                with ui.row().classes('w-full items-center no-wrap px-1 pb-1'):
                    # Opens the Web Upload dialog instead of a native window
                    ui.button(icon='attach_file', on_click=upload_dialog.open).props('flat round dense color=white')
                    text_input = ui.input(placeholder="Type your message...").classes('flex-grow px-2').props('borderless dark')
                    ui.button(icon='send', on_click=lambda: asyncio.create_task(send_message())).props('flat round dense color=primary')
            
            async def send_message():
                user_text = text_input.value.strip()
                if not user_text and not pending_uploads:
                    ui.notify("Please provide some text or an image.", color='warning')
                    return
                    
                payload = []
                images_for_ui = []
                raw_text_record = user_text
                
                if user_text:
                    payload.append(user_text)
                    
                for item in pending_uploads:
                    if item['type'] == 'image':
                        payload.append(item['data'])
                        raw_text_record += f"\n[Uploaded Image: {item['name']}]"
                        buffered = io.BytesIO()
                        item['data'].save(buffered, format="PNG")
                        img_str = base64.b64encode(buffered.getvalue()).decode()
                        images_for_ui.append(f'<img src="data:image/png;base64,{img_str}" />')
                    elif item['type'] == 'text':
                        payload.append(item['data'])
                        raw_text_record += f"\n[Uploaded File: {item['name']}]"

                chat_history.append({
                    'text': user_text if user_text else "📎 *(Attachments sent)*", 
                    'user_input_only': user_text,
                    'name': 'Student', 
                    'sent': True, 
                    'role': 'user',
                    'raw_text': raw_text_record,
                    'images': images_for_ui
                })
                
                text_input.value = ""
                pending_uploads.clear()
                render_previews.refresh()
                render_messages.refresh()
                scroll_area.scroll_to(percent=1)

                current_instruction = build_system_prompt(mode_select.value, language_select.value, course_select.value)
                
                gemini_history = []
                for msg in chat_history[:-1]:
                    role = msg['role']
                    gemini_history.append(types.Content(role=role, parts=[types.Part.from_text(text=msg['raw_text'])]))

                try:
                    chat = client.aio.chats.create(
                        model=MODEL_ID,
                        config=types.GenerateContentConfig(
                            system_instruction=current_instruction,
                            temperature=0.7 if mode_select.value == "Socratic" else 0.2
                        ),
                        history=gemini_history
                    )
                    
                    chat_history.append({'text': '', 'name': 'DACodeX', 'sent': False, 'role': 'model', 'raw_text': ''})
                    render_messages.refresh()
                    scroll_area.scroll_to(percent=1)
                    
                    response_stream = await chat.send_message_stream(payload)
                    full_response = ""
                    
                    async for chunk in response_stream:
                        if chunk.text:
                            full_response += chunk.text
                            
                            # Update the UI with the natural chunks as they arrive
                            chat_history[-1]['text'] = full_response
                            chat_history[-1]['raw_text'] = full_response
                            
                            # Refreshing per chunk is much easier on the network
                            render_messages.refresh()
                            scroll_area.scroll_to(percent=1)
                                
                except Exception as e:
                    ui.notify(f"🤖 Technical Hiccup: {str(e)}", color='negative')

            text_input.on('keydown.enter', send_message)

    def start_interface():
        landing_view.set_visibility(False)
        main_chat_view.set_visibility(True)
        drawer.value = True 
    
    start_btn.on_click(start_interface)

# --- SERVER INITIALIZATION ---
if __name__ in {"__main__", "__mp_main__"}:
    # WEB MODIFICATION: native=False, Host binding to 0.0.0.0, specific port 
    ui.run(
        title="DACodeX - Academic Core", 
        dark=True,
        host='0.0.0.0',  # Listens to external devices on your network
        port=8081,       # The port clients will connect to
        reload=False
    )