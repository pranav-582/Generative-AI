import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)

prompt = "Write a 50 word paragraph on why are trees important"

response = llm.invoke(prompt)

print(response.content)