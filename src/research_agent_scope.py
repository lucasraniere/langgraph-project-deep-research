
"""
User clarification and Research Brief Generation

This module implements the scoping phase of the research workflow, where we:
1. Assess if the user's requests needs clarification
2. Generate a detailed research brief from the conversation

The workflow uses structured output to make deterministic decisions abou
whether sufficient context existis to proceed with research.
"""

from datetime import datetime
from typing_extensions import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, get_buffer_string
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from prompts import clarify_with_user_instructions, transform_messages_into_research_topic_prompt
from states.state_scope import AgentState, ClarifyWithUser, ResearchQuestion, AgentInputState

from langchain_core.rate_limiters import InMemoryRateLimiter

import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# == Utility Functions ==
def get_today_str() -> str:
    """Get current date in a human-readable format."""
    return datetime.now().strftime("%a %b %-d, %Y")


# == Configuration ==

# Initialize model
rate_limiter = InMemoryRateLimiter(
    requests_per_second=8.3,
    check_every_n_seconds=0.1,
    #max_bucket_size=10
)
model = init_chat_model(model="openai:gpt-4.1", temperature=0.0, rate_limiter=rate_limiter)

# Nodes
def clarify_with_user(state: AgentState) -> Command[Literal["write_research_brief", "__end__"]]:
    """
    Clarification decision node.

    Determines if the user's request contains sufficient information to proceed
    with research or if additional clarification is need.

    Uses structured output to make deterministic decisions and avoid hallucination.
    Routes to either research brief generation of ends with a clarification question.
    """
    # Set up structured output model
    structured_output_model = model.with_structured_output(ClarifyWithUser)

    # Invoke the model with clarification instructions
    response = structured_output_model.invoke([
        HumanMessage(content=clarify_with_user_instructions.format(
            messages=get_buffer_string(messages=state["messages"]),
            date=get_today_str()
        ))
    ])

    # Route based on clarification need
    if response.need_clarification:
        return Command(
            goto=END,
            update={"messages": [AIMessage(content=response.question)]}
        )
    else:
        return Command(
            goto="write_research_brief",
            update={"messages": [AIMessage(content=response.verification)]}
        )


def write_research_brief(state: AgentState):
    """
    Research brief generation node.

    Transforms the conversation history into a detailed research brief
    that will guide the subsequent research phase.

    Uses structured output to ensure the brief follow the required format
    and contains all necessary details for effective research.
    """
    # Set up structured output model
    structured_output_model = model.with_structured_output(ResearchQuestion)

    # Generate research brief from conversation history
    response = structured_output_model.invoke([
        HumanMessage(content=transform_messages_into_research_topic_prompt.format(
            messages=get_buffer_string(messages=state["messages"]),
            date=get_today_str()
        ))
    ])

    # Update state with generated research brief and pass it to the supervisor
    return {
        "research_brief": response.research_brief,
        "supervisor_messages": [HumanMessage(content=f"{response.research_brief}.")]
    }

# == Graph Construcion ==

# Building the scoping workflow
deep_researcher_builder = StateGraph(AgentState, input_schema=AgentInputState)

# Add nodes
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
deep_researcher_builder.add_node("write_research_brief", write_research_brief)

# Add edges
deep_researcher_builder.add_edge(START, "clarify_with_user")
deep_researcher_builder.add_edge("write_research_brief", END)

# Compile the workflow
scope_research = deep_researcher_builder.compile()
