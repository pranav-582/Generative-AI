import os
from typing import Dict, Any, List
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import google.generativeai as genai

load_dotenv()

# Configure Google Generative AI for token counting
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
genai_model = genai.GenerativeModel('gemini-1.5-flash')

class DebugCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        self.call_count += 1
        print(f"\n=== LLM START (Call #{self.call_count}) ===")
        print(f"Model: {serialized.get('name', 'Unknown')}")
        
        if prompts:
            try:
                result = genai_model.count_tokens(prompts[0])
                input_tokens = result.total_tokens
            except Exception as e:
                print(f"Token count error: {e}")
                input_tokens = -1
            print(f"Google token count (input): {input_tokens}")
            self.total_input_tokens += input_tokens if input_tokens > 0 else 0
        print("================================")
    
    def on_llm_end(self, response, **kwargs) -> None:
        print(f"\n=== LLM END (Call #{self.call_count}) ===")
        
        response_text = str(response.generations[0][0].text)
        try:
            result = genai_model.count_tokens(response_text)
            output_tokens = result.total_tokens
        except Exception as e:
            print(f"Token count error: {e}")
            output_tokens = -1
        print(f"Google token count (output): {output_tokens}")
        self.total_output_tokens += output_tokens if output_tokens > 0 else 0
        print(f"Total tokens so far: {self.total_input_tokens + self.total_output_tokens}")
        print("===============================")
    
    def get_token_usage(self) -> Dict[str, int]:
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_calls": self.call_count
        }

class SocialMediaAgent:
    def __init__(self):
        self.callback_handler = DebugCallbackHandler()
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            callbacks=[self.callback_handler]
        )

        self.linkedin_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a professional LinkedIn content creator. Create an engaging LinkedIn post 
            based on the job description provided. The post should be professional, engaging, and highlight 
            key skills and opportunities. Keep it under 300 words and include relevant hashtags."""),
            ("user", "Job Description: {job_description}")
        ])
        
        self.instagram_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a creative Instagram content creator. Create an engaging Instagram post 
            based on the job description provided. The post should be casual, visually appealing in text, 
            and use relevant hashtags. Keep it under 250 words and make it inspiring for job seekers."""),
            ("user", "Job Description: {job_description}")
        ])

        self.linkedin_chain = (
            self.linkedin_prompt
            | self.llm
            | StrOutputParser()
        )
        
        self.instagram_chain = (
            self.instagram_prompt
            | self.llm
            | StrOutputParser()
        )
    
    def generate_post(self, job_description: str, platform: str = "linkedin") -> str:
        """Generate social media post using LCEL chains with pipe operators"""
        print(f"\n>>> GENERATING {platform.upper()} POST...")
        
        if platform == "linkedin":
            result = self.linkedin_chain.invoke({"job_description": job_description})
            return f"LinkedIn Post:\n\n{result}"
        elif platform == "instagram":
            result = self.instagram_chain.invoke({"job_description": job_description})
            return f"Instagram Post:\n\n{result}"
        elif platform == "both":
            print(">>> Generating LinkedIn post...")
            linkedin_result = self.linkedin_chain.invoke({"job_description": job_description})
            
            print(">>> Generating Instagram post...")
            instagram_result = self.instagram_chain.invoke({"job_description": job_description})
            
            return f"LinkedIn Post:\n\n{linkedin_result}\n\n{'='*50}\n\nInstagram Post:\n\n{instagram_result}"
        else:  
            result = self.linkedin_chain.invoke({"job_description": job_description})
            return f"LinkedIn Post:\n\n{result}"
    
    def get_token_usage_summary(self) -> None:
        """Display token usage summary"""
        usage = self.callback_handler.get_token_usage()
        print(f"\n{'='*60}")
        print("TOKEN USAGE SUMMARY")
        print(f"{'='*60}")
        print(f"Total LLM calls: {usage['total_calls']}")
        print(f"Total input tokens: {usage['total_input_tokens']}")
        print(f"Total output tokens: {usage['total_output_tokens']}")
        print(f"Total tokens used: {usage['total_tokens']}")
        print(f"{'='*60}")

def get_user_input():
    """Get job description and platform choice from user"""
    print("\n" + "="*60)
    print("SOCIAL MEDIA POST GENERATOR")
    print("="*60)

    print("\nPlease enter the job description:")
    print("(Press Enter twice when finished)")
    job_description_lines = []
    while True:
        line = input()
        if line == "" and job_description_lines:
            break
        job_description_lines.append(line)
    
    job_description = "\n".join(job_description_lines)
    
    if not job_description.strip():
        print("No job description provided. Using default...")
        job_description = """
        Senior Software Engineer - AI/ML
        
        We are looking for a Senior Software Engineer specializing in AI/ML to join our innovative team. 
        You will be responsible for developing and implementing machine learning models, working with 
        large datasets, and building scalable AI solutions.
        
        Requirements:
        - 5+ years of software engineering experience
        - Strong Python programming skills
        - Experience with TensorFlow, PyTorch, or similar ML frameworks
        - Knowledge of cloud platforms (AWS, GCP, Azure)
        - Experience with data pipelines and MLOps
        
        We offer competitive salary, remote work options, and the opportunity to work on cutting-edge AI projects.
        """

    print("\n" + "-"*50)
    print("Choose which platform post to generate:")
    print("1. LinkedIn")
    print("2. Instagram") 
    print("3. Both platforms")
    print("-"*50)
    
    while True:
        choice = input("Enter your choice (1-3): ").strip()
        if choice == "1":
            return job_description, "linkedin"
        elif choice == "2":
            return job_description, "instagram"
        elif choice == "3":
            return job_description, "both"
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

def main():
    print("Initializing Social Media Agent...")
    agent = SocialMediaAgent()
    
    while True:
        try:
            job_description, platform = get_user_input()
            result = agent.generate_post(job_description, platform)
            print(f"\nGenerated Post:")
            print("=" * 60)
            print(result)
            print("=" * 60)

            agent.get_token_usage_summary()
            
        except Exception as e:
            print(f"Error generating post: {str(e)}")

        print("\n" + "-"*50)
        continue_choice = input("Do you want to generate another post? (y/n): ").strip().lower()
        if continue_choice not in ['y', 'yes']:
            print("Thank you for using the Social Media Post Generator!")
            agent.get_token_usage_summary()
            break

if __name__ == "__main__":  
    print("Social Media Post Generator")
    print("==========================")
    main()