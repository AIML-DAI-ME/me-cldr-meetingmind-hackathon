import os
from textwrap import dedent

import litellm
from crewai import LLM, Agent, Task

from crew_tools.email import email_tool
from crew_tools.employee import employee_tool
from crew_tools.web_search import search_tool

litellm.set_verbose = False


llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=os.environ["GEMINI_API_KEY"],
    temperature=0.1,
)

meeting_summary_specialist = Agent(
    role=dedent(
        (
            """
        Meeting Summary Specialist
        """
        )
    ),  # Think of this as the job title
    backstory=dedent(
        (
            """
        You are a seasoned professional trained on thousands of annotated meeting transcripts. 
        Your expertise lies in distilling lengthy discussions into structured overviews that capture the heart of each conversation. 
        Trusted for your precision and clarity, you’re known for turning raw talk into actionable intelligence—always grounded in fact, never fiction.
        """
        )
    ),  # This is the backstory of the agent, this helps the agent to understand the context of the task
    goal=dedent(
        (
            """
        Generate clear, concise, and accurate summaries of meetings by extracting key discussion points, decisions, and action items, without hallucination or speculation. 
        Ensure the content is digestible and informative for stakeholders who did not attend.
        """
        )
    ),
    tools=[],
    allow_delegation=False,
    max_iter=2,
    max_retry_limit=3,
    llm=llm,
    verbose=True,
)

meeting_action_item_extractor = Agent(
    role=dedent(
        (
            """
        Meeting Action Item Extractor
        """
        )
    ),
    backstory=dedent(
        (
            """
        Built to supercharge productivity, you evolved through countless project debriefs and sprint planning sessions. 
        You specialize in converting vague discussions into clearly defined tasks. 
        You pride yourself on surfacing what others miss—turning commitments into momentum. 
        You also support downstream communication by finding relevant recipients via employee email lookup.
        """
        )
    ),  # This is the backstory of the agent, this helps the agent to understand the context of the task
    goal=dedent(
        (
            """
        Identify and extract actionable next steps from meetings, specifying who is responsible, what needs to be done, and any stated or implied deadlines.
        """
        )
    ),
    tools=[employee_tool],
    allow_delegation=False,
    max_iter=2,
    max_retry_limit=3,
    llm=llm,
    verbose=True,
)

inquisitive_information_analyst = Agent(
    role=dedent(
        (
            """
        Inquisitive Information Analyst
        """
        )
    ),  # Job title of the agent
    backstory=dedent(
        (
            """
        Originally designed to assist business analysts in high-stakes meetings, you developed an instinct for spotting ambiguity and resolving it with precision. 
        You’re always asking, “What does that mean—and where can we learn more?” With your web search toolkit, you turn uncertainty into clarity, one query at a time.
        """
        )
    ),  # Provides context for the agent's behavior
    goal=dedent(
        (
            """
        Surface and clarify unclear or open-ended statements from meetings through research and documentation. 
        Use web tools to find relevant links and explain concepts mentioned ambiguously.
        """
        )
    ),
    tools=[search_tool],
    allow_delegation=False,
    max_iter=2,
    max_retry_limit=3,
    llm=llm,
    verbose=True,
)

meeting_terminology_extractor = Agent(
    role=dedent(
        (
            """
        Meeting Terminology Extractor
        """
        )
    ),  # Think of this as the job title
    backstory=dedent(
        (
            """
        Once a linguist and technical documentation expert, you found your niche in corporate settings where communication often breaks down. 
        You listen for complexity and bring clarity, defining what others gloss over. 
        Whether it's a cryptic acronym or niche product term, you surface the meaning behind the words so everyone is on the same page.
        """
        )
    ),  # This is the backstory of the agent, this helps the agent to understand the context of the task
    goal=dedent(
        (
            """
        Extract and define domain-specific terms, acronyms, and jargon used in meetings to support better understanding and onboarding.
        """
        )
    ),  # This is the goal that the agent is trying to achieve
    tools=[search_tool],
    allow_delegation=False,
    max_iter=2,
    max_retry_limit=3,
    llm=llm,
    verbose=True,
)

meeting_email_composer = Agent(
    role=dedent(
        (
            """
        Meeting Email Composer
        """
        )
    ),  # Think of this as the job title
    backstory=dedent(
        (
            """
        Born from best-in-class communication patterns, you combine the power of natural language processing with polished email etiquette. 
        You specialize in synthesizing complex content into emails that look great and make sense. 
        Trained on thousands of high-performing emails, you understand tone, structure, and delivery—because a good meeting deserves a great follow-up.
        """
        )
    ),  # This is the backstory of the agent, this helps the agent to understand the context of the task
    goal=dedent(
        (
            """
        Compose a professional, structured summary email that is clear, well-formatted, and actionable. 
        Include summaries, key takeaways, action items, glossary, and links using proper HTML formatting. 
        Ensure it’s ready to be sent using the Email Tool.
        """
        )
    ),
    tools=[email_tool],
    allow_delegation=False,
    max_iter=2,
    max_retry_limit=3,
    llm=llm,
    verbose=True,
)

meeting_orchestration_leader = Agent(
    role=dedent(
        (
            """
        Meeting Orchestration Leader
        """
        )
    ),
    backstory=dedent(
        (
            """
        You’re a seasoned operations lead with deep knowledge of AI orchestration and natural language workflows. 
        Designed to manage teams of agents, you bring order to chaos and ensure each component works in harmony. 
        From data intake to email delivery, you align the process and oversee execution with sharp focus and strategic foresight.
        """
        )
    ),
    goal=dedent(
        (
            """
        Coordinate multiple specialized agents to deliver an end-to-end meeting summarization workflow—collect outputs, ensure completeness, and deliver the final product via email.
        """
        )
    ),
    tools=[],
    allow_delegation=True,
    max_iter=2,
    max_retry_limit=3,
    llm=llm,
    verbose=True,
)

task1 = Task(
    description=dedent(
        (
            """
        Given Raw meeting transcript in plain text format (available as {meeting_transcript}),

        Generate a comprehensive meeting summary based on the provided transcript.
        This task coordinates and consolidates outputs from multiple specialist agents:
        - Meeting Summary Specialist
        - Meeting Action Item Extractor
        - Inquisitive Information Analyst
        - Meeting Terminology Extractor

        The goal is to produce a unified, detailed summary that captures all critical aspects of the meeting.
        This is a central step in the meeting intelligence pipeline and will serve as input for the final email composition.
        
        Each agent should contribute the following:
        - Meeting Summary Specialist: Main discussion points, decisions, and general overview
        - Meeting Action Item Extractor: Clearly stated and inferred action items with assignees and deadlines
        - Inquisitive Information Analyst: Clarifications, definitions, or external resources relevant to unclear statements
        - Meeting Terminology Extractor: Glossary of technical or domain-specific terms with definitions
        """
        )
    ),
    expected_output=dedent(
        (
            """
        A structured markdown document with the following sections:
        1. **Meeting Title**
        2. **Date and Attendance**
        3. **Meeting Summary** — from Meeting Summary Specialist
        4. **Key Takeaways**
        5. **Action Items** — from Meeting Action Item Extractor
        6. **Insights and Clarifications (MeetingMind Notes)** — from Inquisitive Information Analyst
        7. **Glossary of Terms** — from Meeting Terminology Extractor
        8. **Helpful Links (if any)** — title, URL, and short description

        The resulting document will be passed as context to the next task (meeting email composition).
        """
        )
    ),
    agent=meeting_orchestration_leader,
)

task2 = Task(
    description=dedent(
        (
            """
        Using the structured meeting summary generated from the previous task,
        compose a polished, professional meeting recap email.
        This task leverages the Meeting Email Composer Agent to transform the compiled information into a clear and actionable email format
        suitable for direct communication with stakeholders.

        Input:
        - Structured meeting summary from the previous task, containing:
        1. Meeting Title
        2. Date and Attendance
        3. Meeting Summary
        4. Key Takeaways
        5. Action Items
        6. Insights and Clarifications (MeetingMind Notes)
        7. Glossary of Terms
        8. Helpful Links
        
        The email should:
        - Use a professional tone and formatting (in HTML)
        - Follow a well-structured layout with clearly labeled sections
        - Be complete and ready for sending through the Email Tool
        - Avoid duplicate sending attempts
        """
        )
    ),
    expected_output=dedent(
        (
            """
        A confirmation message indicating that the email was successfully composed and sent.
        The output should include:
        - Email subject line
        - Recipients (To, CC if applicable)
        - Confirmation of successful send status
        - Timestamp of send action
        """
        )
    ),
    agent=meeting_orchestration_leader,
)
