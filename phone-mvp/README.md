### System Overview: Re-sonance

Re-sonance is an AI-powered companion designed to support individuals living with dementia. It combines real-time speech, personalized memory retrieval, and empathetic conversation to create a supportive environment.

### Key Components

    User Interface (Gradio): A dual-tab system that separates the "Patient" view (simple voice interaction) from the "Caregiver" dashboard (management and monitoring).

    Speech-to-Text (STT): Uses the openai/whisper-large-v3-turbo model to process voice input locally with high accuracy.

    LLM (Large Language Model): Uses Ollama with llama3.2 to generate warm, context-aware responses.

    Memory Backbone (RAG): Uses a vector database (ChromaDB) to recall personal patient history, ensuring the AI knows specific details about the patient’s life and needs.

    Text-to-Speech (TTS): A high-fidelity engine that synthesizes the LLM's text into clear, gentle voice output.

### Requirements for Setup

    Software: Python 3.10+, Ollama (running llama3.2).

    Hardware: An NVIDIA GPU is strongly recommended to ensure real-time audio processing and smooth AI responses.

    Libraries: gradio, transformers, torch, pandas, and chromadb.

### How to Run the Application

    Ensure all your script files (app.py, tts_engine.py, rag_memory.py, llm_client.py) are in the same directory.

    Install dependencies via your terminal: pip install gradio transformers torch pandas chromadb.

    Launch the system with: python app.py.

    Open http://localhost:7860 in your web browser.

### Project File Structure

    app.py: Manages the visual layout, user events, and connects the backend services.

    tts_engine.py: Controls the voice synthesis process.

    rag_memory.py: Handles storing and retrieving patient-specific memories.

    llm_client.py: Manages the connection and chat history with the Ollama model.