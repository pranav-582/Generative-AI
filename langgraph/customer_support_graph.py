from dotenv import load_dotenv
from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5)

class DepartmentClassifier(BaseModel):
    department: Literal["billing", "technical", "shipping", "unknown"] = Field(
        ..., description="Department to handle the query."
    )
    escalate: bool = Field(
        ..., description="True if the customer is angry or needs escalation."
    )

class State(TypedDict):
    messages: Annotated[list, add_messages]
    department: str | None
    escalate: bool | None

def classify_department(state: State):
    last_message = state["messages"][-1]
    classifier_llm = model.with_structured_output(DepartmentClassifier)
    result = classifier_llm.invoke([
        {
        "role": "system",
        "content": """Classify the user message:
        - department: 'billing' for payment, refund, invoice, or account charges
        - department: 'technical' for errors, bugs, login, or product issues
        - department: 'shipping' for delivery, tracking, or order status
        - department: 'unknown' if none of the above
        Also, set escalate=True if the user is angry, upset, or requests a manager."""
        },
        {"role": "user", "content": last_message.content}
    ])
    return {"department": result.department, "escalate": result.escalate}

def router(state: State):
    if state.get("escalate"):
        return {"next": "escalation"}
    dept = state.get("department", "unknown")
    return {"next": dept}

def billing_agent(state: State):
    messages = [
        {"role": "system", "content": "You are a billing specialist. Help with payments, refunds, or invoices."},
        *state["messages"]  
    ]
    reply = model.invoke(messages)
    return {"messages": [AIMessage(content=reply.content, additional_kwargs={"from_agent": "Billing"})]}

def technical_agent(state: State):
    messages = [
        {"role": "system", "content": "You are a technical support agent. Help with errors, bugs, or product issues."},
        *state["messages"]  
    ]
    reply = model.invoke(messages)
    return {"messages": [AIMessage(content=reply.content, additional_kwargs={"from_agent": "Technical"})]}

def shipping_agent(state: State):
    messages = [
        {"role": "system", "content": "You are a shipping specialist. Help with delivery, tracking, or order status."},
        *state["messages"]  
    ]
    reply = model.invoke(messages)
    return {"messages": [AIMessage(content=reply.content, additional_kwargs={"from_agent": "Shipping"})]}

def escalation_agent(state: State):
    messages = [
        {"role": "system", "content": "You are a supervisor. The customer is upset or requested escalation. Respond with empathy and authority."},
        *state["messages"]  
    ]
    reply = model.invoke(messages)

    return {"messages": [AIMessage(content=reply.content, additional_kwargs={"from_agent": "Supervisor"})]}

def unknown_agent(state: State):
    messages = [
        {"role": "system", "content": "You are a general support agent. The user's request did not match any department. Ask for clarification."},
        *state["messages"]  
    ]
    reply = model.invoke(messages)
    return {"messages": [AIMessage(content=reply.content, additional_kwargs={"from_agent": "General"})]}

graph_builder = StateGraph(State)

graph_builder.add_node("classifier", classify_department)
graph_builder.add_node("router", router)
graph_builder.add_node("billing", billing_agent)
graph_builder.add_node("technical", technical_agent)
graph_builder.add_node("shipping", shipping_agent)
graph_builder.add_node("escalation", escalation_agent)
graph_builder.add_node("unknown", unknown_agent)

graph_builder.add_edge(START, "classifier")
graph_builder.add_edge("classifier", "router")
graph_builder.add_conditional_edges(
    "router",
    lambda state: state.get("next"),
    {
        "billing": "billing",
        "technical": "technical",
        "shipping": "shipping",
        "escalation": "escalation",
        "unknown": "unknown"
    }
)
graph_builder.add_edge("billing", END)
graph_builder.add_edge("technical", END)
graph_builder.add_edge("shipping", END)
graph_builder.add_edge("escalation", END)
graph_builder.add_edge("unknown", END)

graph = graph_builder.compile()

def run_support_bot():
    state = {"messages": [], "department": None, "escalate": None}
    print("Welcome to Customer Support! (type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        
        state["messages"] = state.get("messages", []) + [
            HumanMessage(content=user_input)
        ]
        
        state = graph.invoke(state)
        
        if state.get("messages") and len(state["messages"]) > 0:
            last_message = state["messages"][-1]

            if hasattr(last_message, 'additional_kwargs'):
                agent = last_message.additional_kwargs.get("from_agent", "Unknown")
                content = last_message.content
            else:
                agent = "Unknown"
                content = str(last_message)
            
            print(f"[{agent} Agent] Assistant: {content}")

if __name__ == "__main__":
    run_support_bot()