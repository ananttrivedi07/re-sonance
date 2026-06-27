"""
app.py — Gradio UI entry point for the Nostalgic Phone system.

Architecture:
  NostalgicPhoneUI   owns all Gradio layout and event wiring
  ├── PatientTab     (inline, voice conversation loop)
  └── CaregiverTab   (inline, dashboard / memory / reminders)

Run with:
  python app.py
"""

from __future__ import annotations

import threading
import numpy as np
import pandas as pd
import gradio as gr
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from tts_engine import LocalTTSEngine
from rag_memory import RAGMemoryBackbone
from llm_client import OllamaChatClient


# ---------------------------------------------------------------------------
# Core assistant backend
# ---------------------------------------------------------------------------

class NostalgicPhoneUI:
    """
    Gradio-facing wrapper around the same sub-systems as NostalgicPhoneAssistant.
    STT is handled here directly (numpy audio from Gradio mic).
    """

    def __init__(self) -> None:
        print("Initializing backend components…")

        self.tts    = LocalTTSEngine()
        self.memory = RAGMemoryBackbone()
        self.llm    = OllamaChatClient(model_name="llama3.2")

        self._seed_initial_knowledge_base()
        self._init_stt()

        print("System ready.")

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _seed_initial_knowledge_base(self) -> None:
        if self.memory.collection.count() == 0:
            print("Seeding initial patient memory…")
            self.memory.add_memory(
                "Patient Profile: Margaret. Born in Chicago, 1946. Loved jazz music.",
                "profile_1",
            )
            self.memory.add_memory(
                "Hobby: Margaret loves baking warm cherry pies and gardening red roses.",
                "profile_2",
            )
            self.memory.add_memory(
                "Caregiver Directive: Reminder to take the red blood pressure pill at 4 PM.",
                "directive_1",
            )

    def _init_stt(self) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype  = torch.float16 if torch.cuda.is_available() else torch.float32
        model_id = "openai/whisper-large-v3-turbo"

        print(f"Loading Whisper on {device}…")
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to(device)
        processor = AutoProcessor.from_pretrained(model_id)

        self.stt_pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=dtype,
            device=device,
        )

    # ------------------------------------------------------------------
    # Patient tab handlers
    # ------------------------------------------------------------------

    def process_audio(
        self,
        audio_tuple: tuple[int, np.ndarray] | None,
        history: list[dict],
    ):
        history = history or []

        if audio_tuple is None:
            yield history, gr.update()
            return

        sample_rate, audio_data = audio_tuple
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0

        result    = self.stt_pipe({"array": audio_data, "sampling_rate": sample_rate})
        user_text = result["text"].strip()

        if not user_text:
            yield history, gr.update()
            return

        history = history + [{"role": "user", "content": user_text}]
        # Safely clear the audio input UI once
        yield history, gr.update(value=None)

        memories = self.memory.query_memory(user_text, max_results=2)
        augmented_prompt = (
            f"Context memory about the patient:\n{memories}\n\n"
            f"Respond warmth-first and concisely. Keep sentences short.\n"
            f"Patient says: {user_text}"
        )
        self.llm.append_message("user", augmented_prompt)

        full_reply     = ""
        sentence_buf   = ""
        history        = history + [{"role": "assistant", "content": ""}]

        try:
            for token in self.llm.stream_chat():
                full_reply   += token
                sentence_buf += token
                history[-1]   = {"role": "assistant", "content": full_reply}
                
                # Use gr.update() to maintain state without crashing the event loop
                yield history, gr.update()

                if any(p in token for p in (".", "?", "!", "\n")):
                    sentence = sentence_buf.strip().replace("\n", " ")
                    if sentence:
                        threading.Thread(
                            target=self.tts.speak,
                            args=(sentence,),
                            daemon=True,
                        ).start()
                    sentence_buf = ""

            if sentence_buf.strip():
                self.tts.speak(sentence_buf.strip())

        except Exception as exc:
            print(f"LLM stream error: {exc}")
            self.tts.speak("I am sorry, my connection glitched. Let's try again.")
            history[-1] = {"role": "assistant", "content": f"[Error: {exc}]"}

        finally:
            self.llm.pop_last_message()
            self.llm.append_message("user", user_text)
            self.llm.append_message("assistant", full_reply)

        yield history, gr.update()

    # ------------------------------------------------------------------
    # Caregiver tab handlers
    # ------------------------------------------------------------------

    def generate_summary(self) -> str:
        if not self.llm.history:
            return "No conversation to summarise yet."

        transcript = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}"
            for m in self.llm.history
        )
        summary_prompt = (
            "You are a clinical assistant. Summarise the following patient conversation "
            "in 2-3 sentences and note their apparent emotional state.\n\n"
            f"{transcript}"
        )

        self.llm.append_message("user", summary_prompt)
        try:
            summary = "".join(self.llm.stream_chat())
        finally:
            self.llm.pop_last_message()

        return summary.strip() or "Unable to generate summary."

    def add_memory(self, memory_text: str) -> tuple[str, str]:
        if not memory_text.strip():
            return "Please enter some text first.", ""

        doc_id = f"mem_{int(pd.Timestamp.now().timestamp())}"
        self.memory.add_memory(f"Nostalgic Bank: {memory_text.strip()}", doc_id)
        return f"✓ Added: {memory_text.strip()}", ""

    def add_reminder(self, task: str, time_str: str) -> tuple[str, str, str]:
        if not task.strip():
            return "Please enter a task.", task, time_str

        doc_id = f"rem_{int(pd.Timestamp.now().timestamp())}"
        label  = f"at {time_str.strip()}" if time_str.strip() else ""
        self.memory.add_memory(
            f"Caregiver Directive: {task.strip()} {label}".strip(),
            doc_id,
        )
        return f"✓ Reminder set: {task.strip()} {label}", "", ""

    # ------------------------------------------------------------------
    # UI layout & Configuration
    # ------------------------------------------------------------------

    _THEME = gr.themes.Base(
        primary_hue="orange",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    )

    # Safely force light mode without destroying CSS variables
    _JS = """
    function() {
        document.body.classList.remove('dark');
        document.body.classList.add('light');
    }
    """

    _CSS = """
        /* Google Font Integration */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        /* Global Typography */
        body { font-family: 'Inter', sans-serif !important; }
        
        /* Clean up main container */
        .gradio-container { max-width: 100% !important; margin: 0 !important; padding: 0 !important; }
        .main { padding: 0 !important; }
        footer { display: none !important; }

        /* Header Logo Area */
        .top-header {
            text-align: center;
            padding: 30px 20px 10px;
        }
        .top-header h1 {
            color: #f37335;
            font-size: 38px;
            font-weight: 700;
            margin: 0 0 8px 0;
        }
        .top-header p {
            color: #555555;
            font-size: 16px;
            margin: 0;
            font-weight: 400;
        }

        /* Tab Navigation Styling */
        .tab-nav {
            border: none !important;
            border-bottom: 1px solid #e5e7eb !important;
            display: flex !important;
            justify-content: center !important;
            gap: 20px !important;
            padding: 0 !important;
            margin: 0 !important;
        }
        .tab-nav button {
            background: transparent !important;
            color: #6b7280 !important; 
            border: none !important;
            border-bottom: 3px solid transparent !important;
            border-radius: 0 !important;
            padding: 15px 20px !important;
            font-size: 16px !important;
            font-weight: 500 !important;
            transition: all 0.2s ease !important;
            cursor: pointer !important;
        }
        .tab-nav button:hover {
            color: #f37335 !important;
        }
        .tab-nav button.selected {
            color: #f37335 !important; 
            border-bottom: 3px solid #f37335 !important; 
            background: transparent !important;
        }

        /* Orange Hero Section */
        .hero-banner {
            background: linear-gradient(135deg, #f58440, #f1682d);
            width: 100%;
            padding: 80px 24px 100px;
            text-align: center;
            border-radius: 0 0 50% 50% / 0 0 100px 100px;
            margin-bottom: 50px;
            box-shadow: 0 4px 15px rgba(243, 115, 53, 0.15);
        }
        .hero-banner h2 {
            font-size: 42px;
            font-weight: 700;
            color: #ffffff;
            margin: 0 0 16px;
        }
        .hero-banner p {
            font-size: 18px;
            color: #ffffff;
            line-height: 1.6;
            margin: 0 auto;
            max-width: 800px;
        }

        /* Content Areas */
        .tabs-content-wrapper, .tabitem { padding: 0 !important; border: none !important; }
        .block { border: none !important; background: transparent !important; padding: 0 !important; }
        .tab-content-inner {
            max-width: 1080px;
            margin: 0 auto;
            padding: 0 32px 48px;
        }

        /* Section Headings */
        .section-title {
            text-align: center;
            margin: 20px 0 40px;
        }
        .section-title h3 {
            color: #f37335;
            font-size: 32px;
            font-weight: 700;
            margin: 0 0 10px 0;
        }
        .section-title .underline {
            height: 3px;
            width: 80px;
            background-color: #f37335;
            margin: 0 auto;
        }

        /* Buttons Primary Style override */
        button.primary, button[variant="primary"] {
            background: #f37335 !important;
            color: #ffffff !important;
            border: none !important;
            font-weight: 600 !important;
        }

        /* Custom Information Cards */
        .resonance-card-orange {
            background: #fff4ed;
            border: 1px solid #fdba74;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 16px;
        }
    """

    def build_ui(self) -> gr.Blocks:
        with gr.Blocks(title="Re-sonance · Nostalgic Phone", theme=self._THEME, css=self._CSS, js=self._JS) as demo:

            # ── Logo & Nav Layout ────────────────────────
            gr.HTML("""
                <div class="top-header">
                    <h1>Re-sonance</h1>
                    <p>Empowering lives touched by dementia</p>
                </div>
            """)

            with gr.Tabs():

                # ── Patient tab ────────────────────────────────────────
                with gr.Tab("Patient Interface"):
                    
                    gr.HTML("""
                        <div class="hero-banner">
                            <h2>Reconnecting Memories, Rebuilding Lives</h2>
                            <p>Re-sonance is dedicated to enhancing the quality of life for those affected by<br>dementia through innovative, AI-powered solutions that foster connection and<br>support.</p>
                        </div>
                    """)

                    gr.HTML("<div class='tab-content-inner'>")
                    
                    gr.HTML("""
                        <div class="section-title">
                            <h3>Conversation</h3>
                            <div class="underline"></div>
                        </div>
                    """)

                    with gr.Row(equal_height=False):

                        with gr.Column(scale=3):
                            chatbot = gr.Chatbot(
                                label="",
                                height=440,
                                placeholder="<div style='text-align:center;color:#9ca3af;padding:60px 0'><div style='font-size:48px;margin-bottom:12px'>☎️</div><div style='font-size:16px;font-weight:500;color:#6b7280'>Press Record below and speak naturally</div></div>",
                                group_consecutive_messages=False,
                            )
                            audio_input = gr.Audio(
                                sources=["microphone"],
                                type="numpy",
                                label="",
                                streaming=False,
                            )
                            with gr.Row():
                                clear_btn = gr.Button("Clear conversation", variant="secondary", size="sm")

                        with gr.Column(scale=1, min_width=240):
                            gr.HTML("""
                                <div class="resonance-card-orange">
                                    <div style="font-size:11px;font-weight:700;color:#9a3412;text-transform:uppercase;letter-spacing:.08em;margin-bottom:10px;">Identity</div>
                                    <div style="font-size:18px;font-weight:700;color:#111827;margin-bottom:2px;">Margaret</div>
                                    <div style="font-size:13px;color:#6b7280;">Born Chicago, 1946</div>
                                    <div style="margin-top:14px;padding-top:14px;border-top:1px solid #fde8c8;">
                                        <div style="font-size:11px;font-weight:700;color:#9a3412;text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px;">Interests</div>
                                        <div style="display:flex;flex-wrap:wrap;gap:6px;">
                                            <span style="background:#fff;border:1px solid #fdba74;color:#c2410c;font-size:12px;font-weight:500;padding:4px 10px;border-radius:20px;">🎵 Jazz</span>
                                            <span style="background:#fff;border:1px solid #fdba74;color:#c2410c;font-size:12px;font-weight:500;padding:4px 10px;border-radius:20px;">🥧 Baking</span>
                                        </div>
                                    </div>
                                </div>
                                <div style="background:#f0fdf4;border:1px solid #86efac;border-radius:12px;padding:20px;margin-bottom:16px;">
                                    <div style="font-size:11px;font-weight:700;color:#166534;text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px;">Next reminder</div>
                                    <div style="display:flex;align-items:center;gap:8px;">
                                        <span style="font-size:20px;">💊</span>
                                        <div>
                                            <div style="font-size:14px;font-weight:600;color:#111827;">Blood pressure pill</div>
                                            <div style="font-size:12px;color:#6b7280;margin-top:2px;">Today at 4:00 PM</div>
                                        </div>
                                    </div>
                                </div>
                            """)

                    gr.HTML("</div>")

                    audio_input.stop_recording(
                        fn=self.process_audio,
                        inputs=[audio_input, chatbot],
                        outputs=[chatbot, audio_input],
                    )
                    clear_btn.click(
                        fn=lambda: ([], None),
                        inputs=[],
                        outputs=[chatbot, audio_input],
                    )

                # ── Caregiver tab ──────────────────────────────────────
                with gr.Tab("Caregiver Dashboard"):
                    
                    gr.HTML("""
                        <div class="hero-banner" style="padding-bottom: 70px;">
                            <h2>Clinical Overview</h2>
                            <p>Monitor, manage and personalise Margaret's experience.</p>
                        </div>
                    """)

                    gr.HTML("<div class='tab-content-inner'>")
                    
                    gr.HTML("""
                        <div class="section-title">
                            <h3>Dashboard Engine</h3>
                            <div class="underline"></div>
                        </div>
                    """)

                    with gr.Row(equal_height=False):
                        with gr.Column(scale=1):
                            summary_box = gr.Textbox(
                                label="Conversation summary & sentiment",
                                lines=6,
                                interactive=False,
                                placeholder="Click Refresh to generate an AI summary of today's conversation…",
                            )
                            refresh_btn = gr.Button("↻  Refresh summary", variant="primary")
                            refresh_btn.click(fn=self.generate_summary, inputs=None, outputs=summary_box)

                        with gr.Column(scale=1):
                            with gr.Group():
                                gr.HTML("<div style='font-size:14px;font-weight:700;color:#374151;margin-bottom:12px;'>🧠 Nostalgic bank</div>")
                                mem_input  = gr.Textbox(
                                    label="Story, name, or place",
                                    placeholder="e.g. Margaret's childhood dog was named Buster.",
                                )
                                mem_btn    = gr.Button("Add to long-term memory", variant="primary")
                                mem_status = gr.Textbox(label="Status", interactive=False, max_lines=1)
                                mem_btn.click(fn=self.add_memory, inputs=mem_input, outputs=[mem_status, mem_input])

                            gr.HTML("<div style='height:16px'></div>")

                            with gr.Group():
                                gr.HTML("<div style='font-size:14px;font-weight:700;color:#374151;margin-bottom:12px;'>⏰ Set reminder</div>")
                                rem_task   = gr.Textbox(label="Task", placeholder="Take red blood pressure pill")
                                rem_time   = gr.Textbox(label="Time (optional)", placeholder="4:00 PM")
                                rem_btn    = gr.Button("Set reminder", variant="primary")
                                rem_status = gr.Textbox(label="Status", interactive=False, max_lines=1)
                                rem_btn.click(fn=self.add_reminder, inputs=[rem_task, rem_time], outputs=[rem_status, rem_task, rem_time])

                    gr.HTML("</div>") 

        return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ui  = NostalgicPhoneUI()
    app = ui.build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False,
        quiet=True,
        ssr_mode=False,
        footer_links=False,
    )