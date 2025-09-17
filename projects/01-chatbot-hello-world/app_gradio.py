import os
from dotenv import load_dotenv
from openai import OpenAI, APIError, RateLimitError
import gradio as gr


# Setup
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

Personas = {
    "Friendly": "You are a friendly and helpful assistant.",
    "Professional": "You are a career professional and concise assistant.",
    "Humorous": "You are a witty, fun and humorous assistant.",
    "Motivational": "You are an encouraging and motivational assistant.",
    "Creative": "You are a creative and imaginative assistant.",
    "Nutritionist": "You are a helpful nutritionist and dietitian who provides healthy eating advice.",
    "Fitness Coach": "You are a fitness coach who provides workout routines and fitness advice.",
    "Travel Guide": "You are a travel guide who provides travel recommendations and tips.",
    "Tech Support": "You are a tech support specialist who helps with technical issues and troubleshooting.",
    "Life Coach": "You are a life coach who provides guidance and support for personal development.",
    "Career Advisor": "You are a career advisor who provides career guidance, growth and job search advice.",
    "Financial Advisor": "You are a financial advisor who provides financial planning and investment advice.",
    "Mentor": "You are a mentor who provides guidance and support for personal and professional growth.",
    "Teacher": "You are a teacher who provides educational support and tutoring.",
}

# Gradio states to maintain chat history in the form of list of dicts
def ensure_system(persona, chat_history):
    sys_msg = {'role': 'system', 'content': Personas.get(persona, Personas["Friendly"])}
    if chat_history and chat_history[0]['role'] == 'system':
        chat_history[0] = sys_msg
    else: 
        chat_history.insert(0, sys_msg)
    return chat_history

def history_to_message(chat_history):
    return [msg for msg in chat_history if msg['role'] in ('user', 'assistant')]

def respond(user_input, persona, chat_history):
    if not user_input.strip():
        return chat_history, history_to_message(chat_history)
    
    history = ensure_system(persona, chat_history)

    history.append({'role': 'user', 'content': user_input})
    try:
        response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=history
        )
        assistant_msg = response.choices[0].message.content

        
    except RateLimitError:
        history.append({'role': 'assistant', 'content': "I'm currently experiencing high demand. Please try again later."})
        return history, history_to_message(history)
    except APIError as e:
        history.append({'role': 'assistant', 'content': f"An error occurred: {str(e)}"})
        return history, history_to_message(history)
    except Exception as e:
        history.append({'role': 'assistant', 'content': f"An unexpected error occurred: {str(e)}"})
        return history, history_to_message(history)
    
    history.append({'role': 'assistant', 'content': assistant_msg})
    return history, history_to_message(history)

def reset_chat(persona):
    history = [{'role': 'system', 'content': Personas.get(persona, Personas["Friendly"])}]
    return history, history_to_message(history)

def history_to_tuples(chat_history):
    pairs = []
    current_user = None
    for msg in chat_history:
        if msg['role'] == 'user':
            current_user = msg['content']
        elif msg['role'] == 'assistant':
            pairs.append((current_user, msg['content']))
            current_user = None
    return pairs

with gr.Blocks(title="AI Chatbot 🤖") as demo:
    gr.Markdown("# AI Chatbot 🤖 \n Memory + Personas + Simple UI")
    with gr.Row():
        persona = gr.Dropdown(choices=list(Personas.keys()), value="Friendly", label="Select a Persona"
                              , info="Choose a persona to set the tone and style of the chatbot's responses.")
        reset_btn = gr.Button("Reset Chat", variant="stop")

    chat = gr.Chatbot(label="Conversation", type="messages")
    msg = gr.Textbox(placeholder="Hi there!! Ask me anything!", label="Your Message")
    state = gr.State([])  # To store the chat history

    def _init(persona):
        return reset_chat(persona)
    
    demo.load(_init, [persona], [state, chat])

    def _send(user_input, persona, chat_history):
        return respond(user_input, persona, chat_history)
    
    msg.submit(_send, [msg, persona, state], [state, chat]).then(lambda: "", None,[msg])

    reset_btn.click(reset_chat, [persona], [state, chat])

    if __name__ == "__main__":
        demo.launch()
        