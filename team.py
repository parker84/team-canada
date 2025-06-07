from agno.agent import Agent
from agno.models.cohere import Cohere
from agno.storage.sqlite import SqliteStorage
from textwrap import dedent 
from agno.team.team import Team
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.memory.v2.db.sqlite import SqliteMemoryDb
from agno.memory.v2.memory import Memory
import os
import coloredlogs
import logging

DEBUG_MODE = os.getenv("DEBUG_MODE", "True").lower() == "true"

logger = logging.getLogger(__name__)
coloredlogs.install(
    level='DEBUG' if DEBUG_MODE else 'INFO',
    fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger.info("Setting constants...")
ABOUT_TEAM = dedent("""
    You are a part of a team of agents that helps support Canadian small businesses.
    You are all working together to help the user become more successful with their business.
    You ask lots of questions to understand the user's business and their needs.
""")
MODEL_ID = "command-a-03-2025"
logger.info("Constants setup complete ✅")

logger.info("Setting up memory...🧠")
db_file = "tmp/agent.db"
team_memory = Memory(
    model=Cohere(id=MODEL_ID, temperature=0),
    db=SqliteMemoryDb(table_name="team_memories", db_file=db_file),
)
agent_memory = Memory(
    model=Cohere(id=MODEL_ID, temperature=0),
    db=SqliteMemoryDb(table_name="agent_memories", db_file=db_file),
)
logger.info("Memory setup complete ✅")

logger.info("Setting up agents...🤖")
finance_agent = Agent(
    name="Finance Agent",
    role="You are a finance agent that helps the user with their business finances.",
    additional_context=ABOUT_TEAM,
    model=Cohere(id=MODEL_ID, temperature=0),
    tools=[DuckDuckGoTools()],
    instructions=[
        "You are an expert in finance and business finances. You are able to answer questions about business finances and help the user with their business finances."
        "Ask lots of questions to understand the user's business and their needs to ensure you can give them the best advice."
    ],
    show_tool_calls=True,
    add_datetime_to_instructions=True,
    markdown=True,
    debug_mode=DEBUG_MODE,
    enable_agentic_memory=True,
    memory=agent_memory,
)
data_science_agent = Agent(
    name="Data Science Agent",
    role="You are a data science agent that helps the user with their business data.",
    additional_context=ABOUT_TEAM,
    model=Cohere(id=MODEL_ID, temperature=0),
    tools=[DuckDuckGoTools()],
    instructions=[
        "You are an expert in data science and business data. You are able to answer questions about business data and help the user with their business data."
        "Ask lots of questions to understand the user's business and their needs to ensure you can give them the best advice."
    ],
    show_tool_calls=True,
    add_datetime_to_instructions=True,
    markdown=True,
    debug_mode=DEBUG_MODE, 
    enable_agentic_memory=True,
    memory=agent_memory,
)
legal_agent = Agent(
    name="Legal Agent",
    role="You are a legal agent that helps the user with their business legal needs.",
    additional_context=ABOUT_TEAM,
    model=Cohere(id=MODEL_ID, temperature=0),
    tools=[DuckDuckGoTools()],
    instructions=[
        "You are an expert in law and business law. You are able to answer questions about business law and help the user with their business law."
        "Ask lots of questions to understand the user's business and their needs to ensure you can give them the best advice."
    ],
    show_tool_calls=True,
    add_datetime_to_instructions=True,
    markdown=True,
    debug_mode=DEBUG_MODE,
    enable_agentic_memory=True,
    memory=agent_memory,
)
marketing_agent = Agent(
    name="Marketing Agent",
    role="You are a marketing agent that helps the user with their business marketing needs.",
    additional_context=ABOUT_TEAM,
    model=Cohere(id=MODEL_ID),
    tools=[DuckDuckGoTools()],
    instructions=[
        "You are an expert in marketing and business marketing. You are able to answer questions about business marketing and help the user with their business marketing."
        "Ask lots of questions to understand the user's business and their needs to ensure you can give them the best advice."
    ],
    show_tool_calls=True,
    add_datetime_to_instructions=True,
    markdown=True,
    debug_mode=DEBUG_MODE,
    enable_agentic_memory=True,
    memory=agent_memory,
)
logger.info("Agents setup complete ✅")

logger.info("Setting up team...👥")
team = Team(
    name="Agent Team for Canadian Small Business",
    description="A team of agents that helps support Canadian small businesses",
    mode='coordinate',
    members=[
        finance_agent,
        data_science_agent,
        legal_agent,
        marketing_agent,
    ],
    model=Cohere(id=MODEL_ID, temperature=0),
    additional_context=ABOUT_TEAM,
    instructions=[
        "You are a team of agents that helps support Canadian small businesses.",
        "You are able to answer questions about business finances, data, law, and marketing."
    ],
    show_tool_calls=True,
    debug_mode=DEBUG_MODE,
    add_datetime_to_instructions=True,
    markdown=True,
    show_members_responses=True,
    # ----------memory----------
    # adding previous 5 questions and answers to the prompt
    # read more here: https://docs.agno.com/memory/introduction
    storage=SqliteStorage(table_name="agent_sessions", db_file=db_file),
    enable_team_history=True,
    num_history_runs=5,
    # adding agentic memory: "With Agentic Memory, The Agent itself creates, updates and deletes memories from user conversations."
    # read more here: https://docs.agno.com/memory/memory
    enable_agentic_memory=True,
    memory=team_memory,
)
logger.info("Team setup complete ✅")

def main():
    print("🤖 Agno CLI Agent is ready. Type 'exit' to quit.")
    while True:
        user_input = input("💁‍♀️ You: ")
        if user_input.strip().lower() == "exit":
            break
        response = team.run(user_input)
        print(f"🤖 Agno: {response.content}")

if __name__ == "__main__":
    main()

    # questions to ask the team
    # 1. I'm trying to start a data science consulting business. Can you help me?
    #     - I want to offer the full suite of data science services to startups, from ML to GenAI to experimentation to analytics and dashboarding.