from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import (
    LLMChain,
    SimpleSequentialChain,
    SequentialChain,
    TransformChain,
    ConversationChain
)
from langchain.memory import ConversationBufferMemory
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
import re

from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document

load_dotenv()

llm = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

# -----------------------------------------------------------------------------------------------------------------------------------------
# Practice Question (LLMChain)
# Scenario:
# A client wants a tool that takes a product description and generates a catchy marketing slogan using an LLMChain.

# Task:
# Write a LangChain LLMChain that:

# Takes a product description as input.
# Uses a prompt to ask the LLM to generate a catchy slogan.
# Prints the slogan.
# -----------------------------------------------------------------------------------------------------------------------------------------

prompt = PromptTemplate(
    input_variables=["description"],
    template="You a profesional Social Media Manager experienced in working and managing social media of big companies. " 
    "What you have to is 'Using the product description given by the user generate a catchy slogan for the product'." 
    " The slogan should be catchy like in terms of the things trending online. "
    "The Product description given by the user is: {description}"
)


chain = LLMChain(llm=llm, prompt=prompt)

product_description = input("Enter the product description: ")

slogan = chain.run(product_description)
print("Slogan", slogan)


# -----------------------------------------------------------------------------------------------------------------------------------------
# Practice Question (SequentialChain)
# Scenario:
# A client wants a workflow that:

# Takes a job description.
# First, summarizes the job description.
# Then, generates a short LinkedIn post based on the summary.
# Task:
# Write a SequentialChain that:

# Accepts a job description as input.
# Summarizes it.
# Generates a LinkedIn post from the summary.
# Prints both the summary and the LinkedIn post.
# -----------------------------------------------------------------------------------------------------------------------------------------

summary_prompt = PromptTemplate(
    input_variables=["job_desc"],
    template = "What you have to is 'Using the job description given by the user summarize it'." 
    "The job description given by the user is: {job_desc}"
)

summary_chain=LLMChain(llm=llm, prompt=summary_prompt, output_key="summary")

linkedin_prompt = PromptTemplate(
    input_variables = ["summary"],
    template = "Write a short LinkedIn post for a job opening based on this summary."
    "The job description summary is: {summary}"
)

linkedin_chain = LLMChain(llm=llm, prompt=linkedin_prompt, output_key="linkedin_post")

seq_chain = SequentialChain(
    chains = [summary_chain, linkedin_chain],
    input_variables=['job_desc'],
    output_variables=["summary","linkedin_post"]
)

job_desc = input("Enter the job description: ")

result = seq_chain({"job_desc": job_desc})
print("Summary:", result["summary"])
print("LinkedIn Post:", result["linkedin_post"])

# -----------------------------------------------------------------------------------------------------------------------------------------
# Practice Question (SimpleSequentialChain)

# Scenario:
# A marketing team wants to quickly process customer reviews. Their workflow is:
# 1. Take a long customer review.
# 2. Distill it into a one-sentence summary.
# 3. Write a polite, professional thank-you tweet based on that summary.

# Task:
# Create a SimpleSequentialChain that automates this two-step process
# -----------------------------------------------------------------------------------------------------------------------------------------

template_summarize = "Summarize this customer review in one sentence:\n\n{review}"
prompt_summarize = PromptTemplate.from_template(template_summarize)
chain_summarize = LLMChain(llm=llm, prompt=prompt_summarize)

template_tweet = "Write a polite thank-you tweet for a customer based on this summary:\n\n{summary}"
prompt_tweet = PromptTemplate.from_template(template_tweet)
chain_tweet = LLMChain(llm=llm, prompt=prompt_tweet)

overall_chain = SimpleSequentialChain(
    chains=[chain_summarize, chain_tweet],
    verbose=True
)

customer_review = (
    "I'm absolutely amazed by the new SuperWidget 5000! The battery life is phenomenal, "
    "lasting me three full days. The setup was a bit tricky, but once I got past that, "
    "the performance has been smooth as silk. The camera quality is also a huge step up."
)
final_tweet = overall_chain.run(customer_review)

print("\n--- Final Tweet ---")
print(final_tweet)

# -----------------------------------------------------------------------------------------------------------------------------------------
# Practice Question (RouterChain)

# Scenario:
# A university's student support bot needs to answer questions about two different topics: "Admissions" and "Campus Events".
# Questions about deadlines and application status should go to the admissions expert.
# Questions about clubs and upcoming festivals should go to the campus events expert.

# Task:
# Create a RouterChain that directs a student's question to the appropriate prompt template.
# -----------------------------------------------------------------------------------------------------------------------------------------


admissions_template = """You are an expert on university admissions. Answer the student's question politely.
Here is the question:
{input}"""

events_template = """You are an expert on campus life and events. Answer the student's question with enthusiasm.
Here is the question:
{input}"""

prompt_infos = [
    {
        "name": "admissions",
        "description": "Good for answering questions about university admissions, applications, and deadlines",
        "prompt_template": admissions_template,
    },
    {
        "name": "campus_events",
        "description": "Good for answering questions about campus events, clubs, and festivals",
        "prompt_template": events_template,
    },
]

destination_chains = {}
for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = PromptTemplate.from_template(template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    destination_chains[name] = chain

router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
    destinations=RouterOutputParser.get_destinations_str(prompt_infos)
)
router_prompt = PromptTemplate.from_template(template=router_template, output_parser=RouterOutputParser())
router_chain = LLMRouterChain.from_llm(llm, router_prompt)

chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=LLMChain(llm=llm, prompt=PromptTemplate.from_template("{input}")),
    verbose=True,
)

print("--- Routing Admissions Question ---")
print(chain.run("What is the application deadline for the fall semester?"))

print("\n--- Routing Events Question ---")
print(chain.run("Are there any music festivals happening on campus next month?"))

# -----------------------------------------------------------------------------------------------------------------------------------------
# Practice Question (TransformChain)

# Scenario:
# An IT department wants to analyze server error logs. The logs are messy and contain timestamps and varying case letters.
# Before sending a log to the LLM for analysis, it must be "cleaned" by:
# 1. Removing the timestamp prefix (e.g., "[2025-08-11 20:00:15] ").
# 2. Converting the entire log to lowercase.

# Task:
# Create a workflow that uses a TransformChain to clean the log before an LLMChain analyzes it.
# -----------------------------------------------------------------------------------------------------------------------------------------

def clean_log_data(inputs: dict) -> dict:
    """Cleans a raw log string."""
    raw_log = inputs["raw_log"]
    no_timestamp_log = re.sub(r"^\[.*?\]\s*", "", raw_log)
    cleaned_log = no_timestamp_log.lower()
    return {"cleaned_log": cleaned_log}

transform_chain = TransformChain(
    input_variables=["raw_log"],
    output_variables=["cleaned_log"],
    transform=clean_log_data
)

analysis_template = "Analyze the following cleaned server log and explain the likely root cause in one sentence:\n\n{cleaned_log}"
analysis_prompt = PromptTemplate.from_template(analysis_template)
analysis_chain = LLMChain(llm=llm, prompt=analysis_prompt, output_key="analysis_result")

workflow_chain = SequentialChain(
    chains=[transform_chain, analysis_chain],
    input_variables=["raw_log"],
    output_variables=["cleaned_log", "analysis_result"], 
    verbose=True
)

messy_log = "[2025-08-11 20:04:15] CRITICAL: Database connection FAILED for user 'admin'. Error Code: 5003."
result = workflow_chain.invoke({"raw_log": messy_log})

print("\n--- Transformation Result ---")
print(f"Cleaned Log: {result['cleaned_log']}")
print(f"Analysis Result: {result['analysis_result']}")

# -----------------------------------------------------------------------------------------------------------------------------------------
# Practice Question (ConversationChain)

# Scenario:
# A client wants a simple customer support chatbot that can have a basic, stateful conversation.
# The bot should remember what the user said previously to answer follow-up questions.

# Task:
# Create a ConversationChain that demonstrates memory over a few interactions.
# -----------------------------------------------------------------------------------------------------------------------------------------

conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory()
)

print("--- Customer Support Chatbot ---")
print("Type 'quit' to exit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit"]:
        print("Bot: Goodbye!")
        break
    response = conversation.predict(input=user_input)
    print("Bot:", response)

# -----------------------------------------------------------------------------------------------------------------------------------------
# Practice Question (MapReduceChain)

# Scenario:
# A legal team needs to get the gist of a very long terms-of-service document.
# The document is too long to fit into the LLM's context window in one go.

# Task:
# Use a MapReduce chain to summarize a long piece of text.
# -----------------------------------------------------------------------------------------------------------------------------------------

from langchain.chains.summarize import load_summarize_chain

long_text = """
The digital age has revolutionized information dissemination, yet it has also introduced significant challenges, particularly concerning misinformation and disinformation. Misinformation, the inadvertent sharing of false information, and disinformation, the deliberate creation and sharing of false information with the intent to deceive, pose threats to social cohesion, democratic processes, and public health. Studies show that false news spreads faster and more broadly than true news, primarily through social media platforms.

Efforts to combat this issue are multifaceted. Technology companies are implementing algorithms to detect and flag false content, though these systems face challenges with context and satire. Fact-checking organizations play a crucial role, but their reach is often limited compared to the viral spread of falsehoods. Media literacy education is considered a long-term solution, aiming to equip individuals with the critical thinking skills necessary to evaluate information sources.

The psychological dimension is also critical. Cognitive biases, such as confirmation bias, make individuals more susceptible to believing information that aligns with their pre-existing beliefs, regardless of its veracity. This creates echo chambers and filter bubbles, reinforcing false narratives within specific communities. Addressing the problem requires a holistic approach that combines technological solutions, regulatory oversight, and public education to foster a more resilient and informed citizenry. Each citizen has a role to play.

Furthermore, the economic incentives driving the spread of false information cannot be ignored. Ad-based revenue models on many digital platforms can inadvertently reward sensationalist and false content because it generates high engagement. Clicks, shares, and comments, regardless of the content's accuracy, translate into revenue. Therefore, any comprehensive strategy must also address these underlying economic structures, potentially through new platform governance models or advertiser pressure for more responsible content policies.
"""

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
docs = [Document(page_content=t) for t in text_splitter.split_text(long_text)]

summary_chain = load_summarize_chain(llm=llm, chain_type="map_reduce", verbose=True)

final_summary = summary_chain.run(docs)

print("\n--- MapReduce Summary ---")
print(final_summary)

