from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
parser = StrOutputParser()

greeting_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are friendly. Give a short greeting."),
    ("user", "Say hello to {name}")
])

compliment_prompt = ChatPromptTemplate.from_messages([
    ("system", "You give nice compliments. Be brief."),
    ("user", "Give a compliment to {name}")
])

advice_prompt = ChatPromptTemplate.from_messages([
    ("system", "You give helpful advice. Keep it short."),
    ("user", "Give advice to {name}")
])

greeting_chain = greeting_prompt | model | parser
compliment_chain = compliment_prompt | model | parser
advice_chain = advice_prompt | model | parser

parallel_chain = RunnableParallel({
    "greeting": greeting_chain,
    "compliment": compliment_chain,
    "advice": advice_chain
})

# result = parallel_chain.invoke({"name": "Pranav"})

# print("Results:")
# print(f"Greeting: {result['greeting']}")
# print(f"Compliment: {result['compliment']}")
# print(f"Advice: {result['advice']}")

inputs = [
    {"name": "Pranav"},
    {"name": "Alex"},
    {"name": "Sam"}
]

results = parallel_chain.batch(inputs)

for i, res in enumerate(results, 1):
    print(f"\n--- Results for input #{i} ---")
    print(f"Greeting: {res['greeting']}")
    print(f"Compliment: {res['compliment']}")
    print(f"Advice: {res['advice']}")
