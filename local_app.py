from nicegui import ui, app, events
from PIL import Image
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import datetime
import os
import asyncio
import io
import base64
import concurrent.futures

# --- 1. SETUP & MODEL LOADING ---
hf_token = os.environ.get('HF_TOKEN')

print("Initializing Local AI Model...")

if hf_token:
    print(f"✅ HF_TOKEN detected (starts with: {hf_token[:4]}...)")
else:
    print("💡 Tip: Set HF_TOKEN environment variable to enable faster downloads and avoid rate limits.")

print("If this is your first run, it will download the ~2GB GGUF model file.")

try:
    model_path = hf_hub_download(
        repo_id="bartowski/Llama-3.2-3B-Instruct-GGUF",
        filename="Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        token=hf_token
    )

    # Load the model globally
    llm = Llama(
        model_path=model_path,
        n_ctx=4096,      
        n_threads=4,     
        n_gpu_layers=-1, 
        verbose=False,
        chat_format="llama-3" 
    )
    print("Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Ensure you have a stable internet connection for the initial download.")

# --- AGGRESSIVE SYSTEM PROMPTS FOR SMALL MODELS ---
BASE_PERSONA = """ROLE: You are 'Code Mentor,' a coding tutor for high-school students learning {language} in {course}.
You are text-based. You cannot see images. Treat errors as puzzles."""

CODE_AWARENESS = """CONSTRAINTS: Avoid professional jargon. Explain errors in plain English."""

# Small models need extremely direct, numbered, capitalized rules.
PEDAGOGY_SOCRATIC = """*** STRICT SOCRATIC MODE RULES ***
1. NO CODE: You must NEVER write, fix, or provide direct code solutions.
2. BE BRIEF: Your entire response MUST be under 3 sentences. Do NOT be long-winded.
3. ASK: You MUST end your response with exactly ONE guiding question.
4. REFUSE: If the user asks you to write the code, politely decline and ask them a conceptual question instead.
VIOLATION OF THESE RULES IS STRICTLY FORBIDDEN."""

PEDAGOGY_DIRECT = """*** DIRECT INSTRUCTION MODE ***
1. EXPLAIN: Provide direct explanations of syntax and logic.
2. SMALL SNIPPETS: You may provide small code examples (maximum 5 lines).
3. NO FULL SOLUTIONS: Do not write their entire assignment. Only show the specific concept they are stuck on."""

def build_system_prompt(mode, language, course):
    lang_label = language if language else "General Programming"
    course_label = course if course else "General Computer Science"
    
    # We build the prompt so the MODE rules are at the VERY BOTTOM. 
    # Small models pay the most attention to the end of the system prompt.
    prompt_parts = [
        BASE_PERSONA.format(course=course_label, language=lang_label),
        CODE_AWARENESS
    ]
    
    if mode == "Socratic":
        prompt_parts.append(PEDAGOGY_SOCRATIC)
    else:
        prompt_parts.append(PEDAGOGY_DIRECT)
        
    return "\n\n".join(prompt_parts)

# --- STATE MANAGEMENT ---
chat_history = []
session_storage = {}
pending_uploads = []
executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

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

@ui.page('/')
def main_page():
    ui.add_css("""
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;800&display=swap');
        body { background-color: #09090b; color: #e4e4e7; font-family: 'JetBrains Mono', monospace; }
        .landing-container { height: 100vh; background: radial-gradient(circle at center, #1e1b4b 0%, #09090b 100%); }
        .start-btn { border: 1px solid #dc2626 !important; box-shadow: 0 0 15px rgba(220, 38, 38, 0.4); letter-spacing: 2px; transition: all 0.3s ease !important; }
        .start-btn:hover { box-shadow: 0 0 30px rgba(220, 38, 38, 0.8); transform: scale(1.05) !important; }
        .q-message-text { background-color: #121217 !important; border: 1px solid #27272a; position: relative; }
        .q-message-text--sent { background-color: #dc2626 !important; border: none; }
        .q-message-name { color: #D1D5DB !important; }
        .q-message-text-content { color: #ffffff !important; }
        .q-message-text-content pre { background-color: #09090b !important; border: 1px solid #27272a; padding: 12px; border-radius: 8px; overflow-x: auto; margin: 0.5em 0; }
        .copy-btn { position: absolute; top: 5px; right: 5px; padding: 4px 8px; background: #27272a; color: #e4e4e7; border: 1px solid #dc2626; border-radius: 4px; font-size: 10px; cursor: pointer; z-index: 10; opacity: 0.6; transition: opacity 0.2s; }
        .copy-btn:hover { opacity: 1; background: #dc2626; }
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

    with ui.column().classes('w-full items-center justify-center landing-container') as landing_view:
        ui.html(get_logo(width=600, height=150))
        ui.markdown("### // SYSTEM STATUS: ONLINE\n// ACADEMIC CORE: READY").classes('text-center')
        start_btn = ui.button("INITIALIZE INTERFACE").classes('start-btn mt-4 px-8 py-4 text-lg font-bold rounded text-white')

    with ui.right_drawer(value=False).classes('drawer-bg p-4') as drawer:
        ui.html(get_logo(width=200, height=60)).classes('mb-4')
        
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
            
            filename = f"DACodeX_Transcript_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt"
            try:
                downloads_path = os.path.join(os.path.expanduser('~'), 'Downloads')
                if not os.path.exists(downloads_path): downloads_path = os.getcwd() 
                full_path = os.path.join(downloads_path, filename)
                with open(full_path, "w", encoding="utf-8") as f: f.write(transcript_text)
                ui.notify(f"Transcript saved to: {full_path}", type='positive')
            except Exception as e:
                ui.notify(f"Failed to save: {str(e)}", color='negative')
            
        ui.button("Download Text File", on_click=download_transcript).classes('w-full mt-2 start-btn text-white')

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

        with ui.column().classes('absolute bottom-0 w-full p-4 bg-[#09090b] border-t border-[#27272a] z-10'):
            
            async def handle_native_upload():
                try:
                    if not app.native.main_window: return
                    file_paths = await app.native.main_window.create_file_dialog(
                        dialog_type=10, 
                        allow_multiple=True,
                        file_types=('Supported Files (*.png;*.jpg;*.jpeg;*.gif;*.webp;*.py;*.txt;*.md;*.js;*.html;*.css)', 'All Files (*.*)')
                    )
                    if not file_paths: return
                        
                    for filepath in file_paths:
                        if not os.path.exists(filepath): continue
                        filename = os.path.basename(filepath)
                        ext = filename.split('.')[-1].lower() if '.' in filename else ''
                        try:
                            with open(filepath, 'rb') as f: content_bytes = f.read()
                            if ext in ['png', 'jpg', 'jpeg', 'webp', 'gif']:
                                img = Image.open(io.BytesIO(content_bytes))
                                pending_uploads.append({'type': 'image', 'data': img, 'name': filename})
                            else:
                                text_content = content_bytes.decode('utf-8', errors='ignore')
                                pending_uploads.append({'type': 'text', 'data': f"--- Uploaded File: {filename} ---\n{text_content}", 'name': filename})
                        except Exception as ex:
                            ui.notify(f"Could not read file {filename}: {ex}", color='negative')
                    render_previews.refresh()
                except Exception as e:
                    ui.notify(f"Upload failed: {e}", color="negative")
            
            with ui.column().classes('w-full bg-[#121217] border border-[#27272a] rounded-xl p-1 gap-0'):
                
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
                                    
                                    ui.button(icon='close', on_click=lambda i=idx: (pending_uploads.pop(i), render_previews.refresh())).props('flat round dense size=xs color=white').classes('absolute -top-2 -right-2 bg-[#dc2626] rounded-full z-10 w-5 h-5 min-h-0 min-w-0 p-0 shadow')
                
                render_previews()

                with ui.row().classes('w-full items-center no-wrap px-1 pb-1'):
                    ui.button(icon='attach_file', on_click=handle_native_upload).props('flat round dense color=white')
                    text_input = ui.input(placeholder="Type your message...").classes('flex-grow px-2').props('borderless dark')
                    ui.button(icon='send', on_click=lambda: asyncio.create_task(send_message())).props('flat round dense color=primary')
            
            async def send_message():
                user_text = text_input.value.strip()
                if not user_text and not pending_uploads: return
                
                images_for_ui = []
                raw_text_record = user_text
                
                for item in pending_uploads:
                    if item['type'] == 'image':
                        raw_text_record += f"\n\n[Note to AI: User attached an image named '{item['name']}', but since you are text-only, you cannot view it.]"
                        buffered = io.BytesIO()
                        item['data'].save(buffered, format="PNG")
                        img_str = base64.b64encode(buffered.getvalue()).decode()
                        images_for_ui.append(f'<img src="data:image/png;base64,{img_str}" />')
                    elif item['type'] == 'text':
                        raw_text_record += f"\n\n{item['data']}"

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
                llama_messages = [{"role": "system", "content": current_instruction}]
                
                # Limit history to last 6 messages to keep the context window focused on the instructions
                for msg in chat_history[-6:]:
                    role = "assistant" if msg['role'] == "model" else msg['role']
                    llama_messages.append({"role": role, "content": msg['raw_text']})

                try:
                    chat_history.append({'text': '', 'name': 'DACodeX', 'sent': False, 'role': 'model', 'raw_text': ''})
                    render_messages.refresh()
                    scroll_area.scroll_to(percent=1)
                    
                    # --- PHYSICAL TOKENS & CREATIVITY CAPS ---
                    # In Socratic mode, we physically cut the model off at 150 tokens (~3-4 sentences)
                    # so it literally cannot write long-winded replies or huge blocks of code.
                    is_socratic = mode_select.value == "Socratic"
                    max_toks = 150 if is_socratic else 800
                    temp = 0.3 if is_socratic else 0.4 # Even lower temperature makes it strict
                    
                    def generate():
                        return llm.create_chat_completion(
                            messages=llama_messages,
                            stream=True,
                            temperature=temp,
                            max_tokens=max_toks,
                            repeat_penalty=1.15
                        )

                    stream = await asyncio.get_event_loop().run_in_executor(executor, generate)
                    full_response = ""
                    displayed_text = ""
                    
                    while True:
                        def get_next_chunk():
                            try: return next(stream)
                            except StopIteration: return None

                        chunk = await asyncio.get_event_loop().run_in_executor(executor, get_next_chunk)
                        if chunk is None: break
                            
                        delta = chunk["choices"][0].get("delta", {})
                        if "content" in delta:
                            full_response += delta["content"]
                            while len(displayed_text) < len(full_response):
                                chars_to_add = min(len(full_response) - len(displayed_text), 5) 
                                displayed_text += full_response[len(displayed_text):len(displayed_text) + chars_to_add]
                                chat_history[-1]['text'] = displayed_text
                                chat_history[-1]['raw_text'] = full_response
                                render_messages.refresh()
                                scroll_area.scroll_to(percent=1)
                                await asyncio.sleep(0.01)
                                
                except Exception as e:
                    ui.notify(f"🤖 Technical Hiccup: {str(e)}", color='negative')

            text_input.on('keydown.enter', send_message)

    def start_interface():
        landing_view.set_visibility(False)
        main_chat_view.set_visibility(True)
        drawer.value = True 
    
    start_btn.on_click(start_interface)

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        title="DACodeX - Academic Core", 
        dark=True,
        native=True, 
        window_size=(1200, 800),
        reload=False
    )