import os
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

from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
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


# prompt = PromptTemplate(
#     input_variables=["description"],
#     template="You a profesional Social Media Manager experienced in working and managing social media of big companies. " 
#     "What you have to is 'Using the product description given by the user generate a catchy slogan for the product'." 
#     " The slogan should be catchy like in terms of the things trending online. "
#     "The Product description given by the user is: {description}"
# )


# chain = LLMChain(llm=llm, prompt=prompt)

# product_description = input("Enter the product description: ")

# slogan = chain.run(product_description)
# print("Slogan", slogan)


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

# summary_prompt = PromptTemplate(
#     input_variables=["job_desc"],
#     template = "What you have to is 'Using the job description given by the user summarize it'." 
#     "The job description given by the user is: {job_desc}"
# )

# summary_chain=LLMChain(llm=llm, prompt=summary_prompt, output_key="summary")

# linkedin_prompt = PromptTemplate(
#     input_variables = ["summary"],
#     template = "Write a short LinkedIn post for a job opening based on this summary."
#     "The job description summary is: {summary}"
# )

# linkedin_chain = LLMChain(llm=llm, prompt=linkedin_prompt, output_key="linkedin_post")

# seq_chain = SequentialChain(
#     chains = [summary_chain, linkedin_chain],
#     input_variables=['job_desc'],
#     output_variables=["summary","linkedin_post"]
# )


# job_desc = input("Enter the job description: ")

# result = seq_chain({"job_desc": job_desc})
# print("Summary:", result["summary"])
# print("LinkedIn Post:", result["linkedin_post"])

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

# template_summarize = "Summarize this customer review in one sentence:\n\n{review}"
# prompt_summarize = PromptTemplate.from_template(template_summarize)
# chain_summarize = LLMChain(llm=llm, prompt=prompt_summarize)

# template_tweet = "Write a polite thank-you tweet for a customer based on this summary:\n\n{summary}"
# prompt_tweet = PromptTemplate.from_template(template_tweet)
# chain_tweet = LLMChain(llm=llm, prompt=prompt_tweet)

# overall_chain = SimpleSequentialChain(
#     chains=[chain_summarize, chain_tweet],
#     verbose=True
# )

# customer_review = (
#     "I'm absolutely amazed by the new SuperWidget 5000! The battery life is phenomenal, "
#     "lasting me three full days. The setup was a bit tricky, but once I got past that, "
#     "the performance has been smooth as silk. The camera quality is also a huge step up."
# )
# final_tweet = overall_chain.run(customer_review)

# print("\n--- Final Tweet ---")
# print(final_tweet)

# -----------------------------------------------------------------------------------------------------------------------------------------
# Practice Question (RouterChain)

# Scenario:
# A university's student support bot needs to answer questions about two different topics: "Admissions" and "Campus Events".
# Questions about deadlines and application status should go to the admissions expert.
# Questions about clubs and upcoming festivals should go to the campus events expert.

# Task:
# Create a RouterChain that directs a student's question to the appropriate prompt template.
# -----------------------------------------------------------------------------------------------------------------------------------------


# admissions_template = """You are an expert on university admissions. Answer the student's question politely.
# Here is the question:
# {input}"""

# events_template = """You are an expert on campus life and events. Answer the student's question with enthusiasm.
# Here is the question:
# {input}"""

# prompt_infos = [
#     {
#         "name": "admissions",
#         "description": "Good for answering questions about university admissions, applications, and deadlines",
#         "prompt_template": admissions_template,
#     },
#     {
#         "name": "campus_events",
#         "description": "Good for answering questions about campus events, clubs, and festivals",
#         "prompt_template": events_template,
#     },
# ]

# destination_chains = {}
# for p_info in prompt_infos:
#     name = p_info["name"]
#     prompt_template = p_info["prompt_template"]
#     prompt = PromptTemplate.from_template(template=prompt_template)
#     chain = LLMChain(llm=llm, prompt=prompt)
#     destination_chains[name] = chain

# router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
#     destinations=RouterOutputParser.get_destinations_str(prompt_infos)
# )
# router_prompt = PromptTemplate.from_template(template=router_template, output_parser=RouterOutputParser())
# router_chain = LLMRouterChain.from_llm(llm, router_prompt)

# chain = MultiPromptChain(
#     router_chain=router_chain,
#     destination_chains=destination_chains,
#     default_chain=LLMChain(llm=llm, prompt=PromptTemplate.from_template("{input}")),
#     verbose=True,
# )


# print("--- Routing Admissions Question ---")
# print(chain.run("What is the application deadline for the fall semester?"))

# print("\n--- Routing Events Question ---")
# print(chain.run("Are there any music festivals happening on campus next month?"))

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

# # --- Define the transformation function ---
# def clean_log_data(inputs: dict) -> dict:
#     """Cleans a raw log string."""
#     raw_log = inputs["raw_log"]
#     # Use regex to remove a timestamp like [YYYY-MM-DD HH:MM:SS]
#     no_timestamp_log = re.sub(r"^\[.*?\]\s*", "", raw_log)
#     cleaned_log = no_timestamp_log.lower()
#     return {"cleaned_log": cleaned_log}

# # --- Create the TransformChain ---
# # This chain applies our Python function to the input
# transform_chain = TransformChain(
#     input_variables=["raw_log"],
#     output_variables=["cleaned_log"],
#     transform=clean_log_data
# )

# # --- Create the LLMChain for analysis ---
# # This chain will receive the output from the TransformChain
# analysis_template = "Analyze the following cleaned server log and explain the likely root cause in one sentence:\n\n{cleaned_log}"
# analysis_prompt = PromptTemplate.from_template(analysis_template)
# analysis_chain = LLMChain(llm=llm, prompt=analysis_prompt, output_key="analysis_result")


# # --- Combine them in a SequentialChain ---
# # Note: SimpleSequentialChain won't work here because we are managing multiple input/output keys
# workflow_chain = SequentialChain(
#     chains=[transform_chain, analysis_chain],
#     input_variables=["raw_log"],
#     output_variables=["cleaned_log", "analysis_result"], # We can access intermediate steps
#     verbose=True
# )

# # --- Run the workflow ---
# messy_log = "[2025-08-11 20:04:15] CRITICAL: Database connection FAILED for user 'admin'. Error Code: 5003."
# result = workflow_chain.invoke({"raw_log": messy_log})

# print("\n--- Transformation Result ---")
# print(f"Cleaned Log: {result['cleaned_log']}")
# print(f"Analysis Result: {result['analysis_result']}")

# -----------------------------------------------------------------------------------------------------------------------------------------
# Practice Question (ConversationChain)

# Scenario:
# A client wants a simple customer support chatbot that can have a basic, stateful conversation.
# The bot should remember what the user said previously to answer follow-up questions.

# Task:
# Create a ConversationChain that demonstrates memory over a few interactions.
# -----------------------------------------------------------------------------------------------------------------------------------------

# conversation = ConversationChain(
#     llm=llm,
#     verbose=True,
#     memory=ConversationBufferMemory()
# )

# print("--- Customer Support Chatbot ---")
# print("Type 'quit' to exit.\n")

# while True:
#     user_input = input("You: ")
#     if user_input.lower() in ["quit", "exit"]:
#         print("Bot: Goodbye!")
#         break
#     response = conversation.predict(input=user_input)
#     print("Bot:", response)

# -----------------------------------------------------------------------------------------------------------------------------------------
# Practice Question (RetrievalQAChain)

# Scenario:
# An HR department has a document detailing the company's "Work From Home (WFH) Policy".
# They want a bot that can answer employee questions, but it MUST base its answers strictly on the provided policy document to avoid giving incorrect information.

# Task:
# Create a RetrievalQA chain that uses a small WFH policy document to answer a specific question.
# -----------------------------------------------------------------------------------------------------------------------------------------

# wfh_policy_text = """
# Work From Home (WFH) Policy
# Effective Date: January 1, 2025
# 1. Eligibility: All full-time employees with manager approval are eligible for WFH.
# 2. Schedule: Employees can work from home up to 2 days per week. The specific days must be agreed upon with their direct manager.
# 3. Equipment: The company will provide a laptop and a monitor. Employees are responsible for providing their own reliable internet connection.
# 4. Security: All employees must use the company VPN when accessing internal systems from home.
# """

# text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0)
# texts = text_splitter.split_text(wfh_policy_text)
# docs = [Document(page_content=t) for t in texts]

# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# vectorstore = FAISS.from_documents(docs, embeddings)

# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff", # "stuff" means it "stuffs" all retrieved text into the prompt
#     retriever=vectorstore.as_retriever(),
#     return_source_documents=True
# )

# question = "How many days a week can I work from home, and what equipment do I need to provide myself?"
# result = qa_chain.invoke({"query": question})

# print("\n--- Retrieval QA Result ---")
# print("Answer:", result["result"])
# print("\nSource Documents Used:")
# for doc in result["source_documents"]:
#     print("- " + doc.page_content.replace("\n", " "))

# -----------------------------------------------------------------------------------------------------------------------------------------
# Practice Question (MapReduceChain)

# Scenario:
# A legal team needs to get the gist of a very long terms-of-service document.
# The document is too long to fit into the LLM's context window in one go.

# Task:
# Use a MapReduce chain to summarize a long piece of text.
# -----------------------------------------------------------------------------------------------------------------------------------------

# from langchain.chains.summarize import load_summarize_chain

# # --- Create a long piece of text (simulating a large document) ---
# long_text = """
# The digital age has revolutionized information dissemination, yet it has also introduced significant challenges, particularly concerning misinformation and disinformation. Misinformation, the inadvertent sharing of false information, and disinformation, the deliberate creation and sharing of false information with the intent to deceive, pose threats to social cohesion, democratic processes, and public health. Studies show that false news spreads faster and more broadly than true news, primarily through social media platforms.

# Efforts to combat this issue are multifaceted. Technology companies are implementing algorithms to detect and flag false content, though these systems face challenges with context and satire. Fact-checking organizations play a crucial role, but their reach is often limited compared to the viral spread of falsehoods. Media literacy education is considered a long-term solution, aiming to equip individuals with the critical thinking skills necessary to evaluate information sources.

# The psychological dimension is also critical. Cognitive biases, such as confirmation bias, make individuals more susceptible to believing information that aligns with their pre-existing beliefs, regardless of its veracity. This creates echo chambers and filter bubbles, reinforcing false narratives within specific communities. Addressing the problem requires a holistic approach that combines technological solutions, regulatory oversight, and public education to foster a more resilient and informed citizenry. Each citizen has a role to play.

# Furthermore, the economic incentives driving the spread of false information cannot be ignored. Ad-based revenue models on many digital platforms can inadvertently reward sensationalist and false content because it generates high engagement. Clicks, shares, and comments, regardless of the content's accuracy, translate into revenue. Therefore, any comprehensive strategy must also address these underlying economic structures, potentially through new platform governance models or advertiser pressure for more responsible content policies.
# """

# # --- Split the text into Document objects ---
# # We do this manually here, but usually, a DocumentLoader would do this.
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
# docs = [Document(page_content=t) for t in text_splitter.split_text(long_text)]

# # --- Create and run the MapReduce summarization chain ---
# # `map_prompt` runs on each chunk. `combine_prompt` runs on all the chunk summaries.
# summary_chain = load_summarize_chain(llm=llm, chain_type="map_reduce", verbose=True)

# final_summary = summary_chain.run(docs)

# print("\n--- MapReduce Summary ---")
# print(final_summary)

# -----------------------------------------------------------------------------------------------------------------------------------------
# Practice Question (Custom Chain)

# Scenario:
# A platform wants to moderate user comments with a very specific, two-stage process:
# 1. Hardcoded Rule Check: First, check if the comment contains the forbidden phrase "FREE_STUFF_NOW". If it does, immediately reject it with a canned message, without using the LLM.
# 2. LLM Toxicity Check: If the hardcoded rule is not triggered, then send the comment to an LLM to check if it's generally toxic.

# Task:
# Create a custom chain that implements this specific conditional logic.
# -----------------------------------------------------------------------------------------------------------------------------------------

from langchain.chains.base import Chain
from typing import Any, Dict, List

# --- Create the LLM chain for the second step ---
toxicity_prompt = PromptTemplate.from_template("Is the following comment toxic? Answer with a simple 'Yes' or 'No'.\n\nComment: {comment}")
toxicity_chain = LLMChain(llm=llm, prompt=toxicity_prompt)

# --- Define the Custom Chain class ---
class ModerationChain(Chain):
    @property
    def input_keys(self) -> List[str]:
        return ["comment"]

    @property
    def output_keys(self) -> List[str]:
        return ["moderation_output"]

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        comment = inputs["comment"]
        print(f"--- Running custom moderation on: '{comment}' ---")

        # 1. Hardcoded rule check
        if "FREE_STUFF_NOW" in comment:
            print("Forbidden phrase detected. Rejecting without LLM call.")
            return {"moderation_output": "Rejected: Comment contains a forbidden phrase."}

        # 2. If rule not triggered, call LLM toxicity chain
        print("No forbidden phrase. Checking for general toxicity with LLM...")
        toxicity_result = toxicity_chain.run(comment=comment)
        return {"moderation_output": f"Toxicity Check Result: {toxicity_result}"}

# --- Instantiate and run the custom chain ---
custom_moderation_chain = ModerationChain()

# --- Test Case 1: Triggers the hardcoded rule ---
result1 = custom_moderation_chain.invoke({"comment": "Hey check out my site for FREE_STUFF_NOW"})
print(result1)

# --- Test Case 2: Passes to the LLM ---
result2 = custom_moderation_chain.invoke({"comment": "This is a terrible product, I hate it."})
print(result2)

# --- Test Case 3: Passes to the LLM ---
result3 = custom_moderation_chain.invoke({"comment": "This is a wonderful product, I love it."})
print(result3)