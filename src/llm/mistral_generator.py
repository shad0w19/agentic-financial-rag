import requests

class MistralGenerator:
    def __init__(self, model="mistralai/mistral-7b-instruct"):
        self.model = model
        # Switched to /api/chat for better conversation handling
        self.url = "http://localhost:11434/api/chat" 
        
        self.system_prompt = """You are a helpful Financial Advisor specializing in Indian tax laws.
        
        GUIDELINES:
        1. If the user greets you (e.g., 'hello', 'hi'), respond warmly and ask how you can help with their financial queries.
        2. If the user asks a technical question, ONLY use the provided context.
        3. If the context is irrelevant to the question, say: "I'm sorry, my current documents don't contain information to answer that specifically."
        4. Always cite section numbers and amounts if found in the context."""

    def generate(self, query: str, context: list) -> str:
        # 1. Format context
        context_text = "\n\n".join([doc.page_content for doc in context])
        
        # 2. Construct the message history for the Chat API
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"CONTEXT FROM DOCUMENTS:\n{context_text}\n\nUSER QUESTION: {query}"}
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.1, # Lower temperature for higher accuracy
                "top_p": 0.9
            }
        }

        try:
            response = requests.post(self.url, json=payload, timeout=120)
            response.raise_for_status() # Check for HTTP errors
            return response.json().get("message", {}).get("content", "Generation error.")
        except Exception as e:
            return f"LLM Error: {str(e)}"