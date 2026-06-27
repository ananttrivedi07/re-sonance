import ollama
from typing import Generator

class OllamaChatClient:
    def __init__(self, model_name="llama3.2"):
        self.model_name = model_name
        self.history = []

    def append_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})

    def pop_last_message(self):
        if self.history:
            self.history.pop()

    def stream_chat(self) -> Generator[str, None, None]:
        """Triggers local model iteration, generating tokens dynamically via generator loops."""
        response_stream = ollama.chat(
            model=self.model_name,
            messages=self.history,
            stream=True
        )
        for chunk in response_stream:
            yield chunk["message"]["content"]