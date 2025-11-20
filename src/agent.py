import os
from typing import List

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from retrieval import ResumeRetriever

# Load environment variables
load_dotenv()

def setup_llm():
    """Sets up the LLM, preferring Google, falling back to OpenAI."""
    if os.getenv("GOOGLE_API_KEY"):
        return ChatGoogleGenerativeAI(model="gemini-1.0-pro", convert_system_message_to_human=True)
    elif os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(model="gpt-3.5-turbo")
    else:
        raise ValueError("No API key found for Google or OpenAI. Please set GOOGLE_API_KEY or OPENAI_API_KEY.")

def search_resumes(query: str) -> str:
    """Searches the resume database for candidates matching the query."""
    retriever = ResumeRetriever()
    results = retriever.retrieve(query, top_k=10, top_n=5)
    
    if not results:
        return "No matching resumes found."
        
    formatted_results = ""
    for i, res in enumerate(results):
        formatted_results += f"Candidate {i+1}:\n"
        formatted_results += f"Source: {res['metadata'].get('source')}\n"
        formatted_results += f"Content: {res['metadata'].get('text')}\n"
        formatted_results += "---\n"
        
    return formatted_results

def create_agent():
    llm = setup_llm()
    
    tools = [
        Tool(
            name="SearchResumes",
            func=search_resumes,
            description="Useful for finding candidates with specific skills or experience. Input should be a search query describing the candidate."
        )
    ]
    
    template = '''Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}'''

    prompt = PromptTemplate.from_template(template)
    
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    
    return agent_executor

if __name__ == "__main__":
    try:
        agent = create_agent()
        
        print("--- Resume Agent Initialized ---")
        query = "Find a senior accountant with audit experience and summarize their skills."
        print(f"User: {query}")
        
        response = agent.invoke({"input": query})
        print(f"Agent: {response['output']}")
                
    except Exception as e:
        print(f"Failed to initialize agent: {e}")
