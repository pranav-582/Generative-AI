from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
parser = StrOutputParser()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a fun animal expert. Give a short, interesting fact."),
    ("user", "Tell me a fun fact about a {animal}.")
])

chain = prompt | model | parser

animals = [
    {"animal": "cat"},
    {"animal": "dog"},
    {"animal": "elephant"},
    {"animal": "penguin"}
]

results = chain.batch(animals)


for animal, fact in zip(animals, results):
    print(f"{animal['animal'].capitalize()}: {fact}")