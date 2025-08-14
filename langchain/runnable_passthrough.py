from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
parser = StrOutputParser()

summarize_prompt = ChatPromptTemplate.from_template("Summarize this text: {text}")
sentiment_prompt = ChatPromptTemplate.from_template("What's the sentiment of: {text}")

summarize_chain = summarize_prompt | model | parser
sentiment_chain = sentiment_prompt | model | parser

smart_chain = RunnableParallel({
    "original": RunnablePassthrough(),      
    "summary": summarize_chain,       
    "sentiment": sentiment_chain         
})

test_input = {"text": "Artificial Intelligence (AI) has rapidly transformed the landscape of technology and society over the past decade. From virtual assistants like Siri and Alexa to advanced recommendation systems on platforms such as Netflix and Amazon, AI is now deeply embedded in our daily lives. One of the most significant breakthroughs in recent years has been the development of large language models, such as OpenAI’s GPT series and Google’s Gemini, which are capable of understanding and generating human-like text.These models have revolutionized fields ranging from customer support to creative writing, enabling businesses to automate responses, generate content, and even assist in complex problem-solving. However, the rise of AI also brings challenges, including concerns about data privacy, ethical use, and the potential for job displacement. As AI systems become more sophisticated, it is crucial for policymakers, technologists, and the public to work together to ensure that these technologies are developed and deployed responsibly. Education and upskilling are becoming increasingly important as AI changes the nature of work. Many experts believe that while AI will automate certain tasks, it will also create new opportunities for innovation and employment in areas that require human creativity, empathy, and critical thinking. In conclusion, the future of AI holds immense promise, but it also demands careful consideration of its societal impacts to maximize benefits and minimize risks."}

result = smart_chain.invoke(test_input)

print(f"Original: {result['original']}\n")
print(f"Summary: {result['summary']}\n")
print(f"Sentiment: {result['sentiment']}")