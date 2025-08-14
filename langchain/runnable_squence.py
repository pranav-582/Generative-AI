from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)

conversation_history = []

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a polite customer support assistant. Use conversation history for context."),
    ("user", "{full_conversation}")
])

parser = StrOutputParser()
chain = prompt | model | parser

print("Customer Support Chat - Type 'quit' to exit")
print("-" * 40)

while True:
    user_input = input("You: ")
    
    if user_input.lower() in ["quit", "exit"]:
        print("Chatbot: Goodbye!")
        break

    full_conversation = ""
    for turn in conversation_history:
        full_conversation += f"User: {turn['user']}\nChatbot: {turn['chatbot']}\n\n"
    full_conversation += f"User: {user_input}"

    response = chain.invoke({"full_conversation": full_conversation})
    print(f"Chatbot: {response}")

    conversation_history.append({
        "user": user_input,
        "chatbot": response
    })
