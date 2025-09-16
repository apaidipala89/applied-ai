from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def main():
    print("AI Chatbot 🤖 (type 'quit' to exit)")
    while True:
        user = input("You: ")
        if user.lower() in ('quit', 'exit', 'bye'):
            print('Bot: Goodby! 👋')
            break
        response = client.chat.completions.create(
            model: 'gpt-4o-mini',
            messages=[{'role': 'user', 'content': user}]
        )
        print('Bot:', response.choices[0].message.content)

if __name__ == "__main__":
    main()


