from langchain_core.runnables import RunnableBranch
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)
parser = StrOutputParser()

order_prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant. The user wants to know about their order status. Respond politely."
)
returns_prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant. The user wants to return or refund an item. Explain the return process."
)
escalate_prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant. The user's request needs a human agent. Apologize and say you'll connect them."
)

parser = StrOutputParser()

order_chain = order_prompt | model | parser
returns_chain = returns_prompt | model | parser
escalate_chain = escalate_prompt | model | parser

branch = RunnableBranch(
    (lambda x: "order" in x["user_input"].lower(), order_chain),
    (lambda x: "return" in x["user_input"].lower() or "refund" in x["user_input"].lower(), returns_chain),
    escalate_chain  
)

# ...existing code...

# def is_order_query(x):
#     return "order" in x["user_input"].lower()

# def is_return_or_refund_query(x):
#     text = x["user_input"].lower()
#     return "return" in text or "refund" in text

# branch = RunnableBranch(
#     (is_order_query, order_chain),
#     (is_return_or_refund_query, returns_chain),
#     escalate_chain  
# )

user_input = input("You: ")
result = branch.invoke({"user_input": user_input})
print(f"Chatbot: {result}\n")
