from typing import Iterator
from langchain_core.runnables import RunnableGenerator
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import time

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
parser = StrOutputParser()

def customer_response_generator(input_stream) -> Iterator[str]:
    input_data = next(iter(input_stream))
    
    customer_issue = input_data.get("issue", "general inquiry")
    customer_name = input_data.get("name", "Customer")
    urgency = input_data.get("urgency", "medium")
    
    if urgency == "high":
        prompt_template = """
        You are a senior customer support specialist. A customer named {name} has a {urgency} priority issue.
        Issue: {issue}
        
        Provide a professional, empathetic, and solution-focused response. Be thorough but concise.
        Start with acknowledgment, then provide solution steps, and end with follow-up offer.
        """
    else:
        prompt_template = """
        You are a helpful customer support agent. A customer named {name} needs assistance.
        Issue: {issue}
        
        Provide a friendly, helpful response with clear solution steps.
        """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | model | parser
    
    response = chain.invoke({
        "name": customer_name, 
        "issue": customer_issue,
        "urgency": urgency
    })

    sentences = response.split('.')
    
    for sentence in sentences:
        if sentence.strip():
            yield sentence.strip() + ". "

support_response_gen = RunnableGenerator(customer_response_generator)

print("Customer Support Response Generator Test")

ticket = {
    "name": "John Doe",
    "issue": "I can't login to my account",
    "urgency": "medium"
}

print(f"Input: {ticket}")
print("\nGenerating response...")

result = support_response_gen.invoke(ticket)

print("Response:")
for sentence in result:
    print(sentence, end="", flush=True)  
    time.sleep(0.01) 


