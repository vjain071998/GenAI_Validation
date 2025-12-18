import ollama

response = ollama.chat(model="llama3", messages=[
    {"role": "user", "content": "Hello, what can you do?"}
])

print(response["message"]["content"])
