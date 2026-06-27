# main.py
import time
from audio_stt import AudioSTTEngine
from tts_engine import LocalTTSEngine
from rag_memory import RAGMemoryBackbone
from llm_client import OllamaChatClient

class NostalgicPhoneAssistant:
    def __init__(self):
        self.stt = AudioSTTEngine()
        self.tts = LocalTTSEngine()
        self.memory = RAGMemoryBackbone()
        self.llm = OllamaChatClient(model_name="llama3.2")
        
        self._seed_initial_knowledge_base()

    def _seed_initial_knowledge_base(self):
        if self.memory.collection.count() == 0:
            print("Seeding initial patient memory structure...")
            self.memory.add_memory("Patient Profile: Margaret. Born in Chicago, 1946. Loved jazz music.", "profile_1")
            self.memory.add_memory("Hobby: Margaret loves baking warm cherry pies and gardening red roses.", "profile_2")
            self.memory.add_memory("Caregiver Directive: Reminder to take the red blood pressure pill at 4 PM.", "directive_1")

    def run(self):
        print("\n=======================================================")
        print("  NOSTALGIC PHONE VOICE CORE RUNNING (Fully Local)   ")
        print("=======================================================")
        
        while True:
            user_text = self.stt.listen_and_transcribe()
            if not user_text:
                continue
                
            print(f"\nUser Said: {user_text}")

            if user_text.lower().strip().rstrip('.') in ['exit', 'goodbye', 'bye']:
                self.tts.speak("Goodbye.")
                break

            matched_memories = self.memory.query_memory(user_text, max_results=2)
            
            augmented_prompt = f"""
            Context Memory about the patient:
            {matched_memories}

            Respond warmth-first and concisely to the patient's request. Keep sentences short.
            Patient says: {user_text}
            """
            
            self.llm.append_message("user", augmented_prompt)
            print("AI Responding: ", end="", flush=True)

            full_reply = ""
            sentence_buffer = ""
            
            try:
                for token in self.llm.stream_chat():
                    print(token, end="", flush=True)
                    full_reply += token
                    sentence_buffer += token

                    # FIX 2: Check for space after punctuation to ensure it's an actual clause ending
                    if any(p in token for p in ['.', '?', '!', '\n']):
                        clean_sentence = sentence_buffer.strip().replace('\n', ' ')
                        if clean_sentence:
                            self.tts.speak(clean_sentence)
                        sentence_buffer = ""

                if sentence_buffer.strip():
                    self.tts.speak(sentence_buffer.strip())
                print() 

                # Context management rotation
                self.llm.pop_last_message()
                self.llm.append_message("user", user_text)
                self.llm.append_message("assistant", full_reply)

            except Exception as e:
                print(f"\nLoop Failure: {e}")
                self.llm.pop_last_message()
                self.tts.speak("I am sorry, my internal connection glitched. Let's try again.")
                time.sleep(1)

if __name__ == "__main__":
    assistant = NostalgicPhoneAssistant()
    assistant.run()