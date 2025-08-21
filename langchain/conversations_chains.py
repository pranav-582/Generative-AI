from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import ConversationSummaryMemory
from langchain.memory import CombinedMemory
from langchain.chains import ConversationChain
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# memory = ConversationBufferMemory(return_messages=True)
# memory = ConversationBufferWindowMemory(k=3, return_messages=True)
# memory = ConversationSummaryMemory(llm=llm, return_messages=True)
memory = CombinedMemory(
    memories=[
        ConversationBufferMemory(return_messages=True),
        ConversationSummaryMemory(llm=llm, return_messages=True)
    ]
)

# ConversationChain
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break
    response = conversation.predict(input=user_input)
    print("Assistant:", response)

