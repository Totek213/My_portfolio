import openai

# Set your OpenAI API key here
openai.api_key = "your-api-key-here"

def chat_with_gpt(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",  # Or use "gpt-3.5-turbo" if you prefer
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=200
    )

    reply = response['choices'][0]['message']['content']
    return reply.strip()

# Simple CLI loop
if __name__ == "__main__":
    print("🤖 GPT Chatbot (type 'exit' to quit)\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        response = chat_with_gpt(user_input)
        print(f"Bot: {response}\n")
