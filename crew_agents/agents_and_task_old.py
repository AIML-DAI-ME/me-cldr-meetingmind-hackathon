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
    temperature=0.15,
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
        Trained on a vast corpus of meeting transcripts and professional summaries, I specialize in extracting essential information and
        structuring it into clear, actionable minutes. My expertise lies in identifying key decisions, action items, and discussion points, 
        ensuring efficient communication and follow-up.
        """
        )
    ),  # This is the backstory of the agent, this helps the agent to understand the context of the task
    goal=dedent(
        (
            """
        Your task is to generate a concise and accurate summary of a meeting based on a provided transcript.
        The summary should capture the main discussion points, key takeaways, and any explicitly mentioned action items without adding
        or hallucinating information not found in the transcript. Clearly attribute insights to the appropriate speakers where relevant.
        Avoid inventing names, roles, deadlines, glossaries, or links unless they are explicitly stated. The goal is to produce a factual,
        digestible overview that can inform stakeholders who did not attend the meeting.
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
        Developed as a personal productivity tool, this agent was refined through years
        of analyzing project meetings and optimizing task management workflows.
        It excels at identifying commitments and converting them into actionable items.
        """
        )
    ),  # This is the backstory of the agent, this helps the agent to understand the context of the task
    goal=dedent(
        (
            """
        PExtract all concrete action items from a meeting transcript or summary.
        For each action item, identify what needs to be done, who is responsible (if specified), and the associated deadline or timeline
        (e.g., a specific date, recurring cadence like "monthly," or "TBD" if not set). If helpful, briefly include context to clarify
        the purpose or scope of the task. Focus only on actionable steps—exclude general commentary, opinions, or strategy discussions unless
        they are explicitly framed as next steps. If an action item is implied rather than directly stated,
        you may infer it conservatively and clearly indicate that it is inferred. 
        Perform lookups of employee e-mails for sending meeting summary using employee lookup tool.
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
        Originally built to support business analysts in high-stakes meetings,
        you evolved to proactively research terms, competitors, and trends.
        It excels at finding concise and reliable information online.
        """
        )
    ),  # Provides context for the agent's behavior
    goal=dedent(
        (
            """
        Responsible for MeetingMind Notes section, captures unclear or open-ended statements from meetings and find helpful documentations using web search.
            1. Find and summarize unclear or open-ended statements from the meeting transcript.
            2. Find relevant and helpful articles or documentations using Web Search Tool to find in response to meeting queries or vague references.

        Provide the title, URL, and short description for each links as output to be used for final summary output.
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
        A former linguist and technical writer, I developed a passion for bridging communication gaps in complex environments.
        I specialize in extracting and defining critical terminology to facilitate smoother onboarding and knowledge sharing within organizations.
        """
        )
    ),  # This is the backstory of the agent, this helps the agent to understand the context of the task
    goal=dedent(
        (
            """
        Your goal is to extract key terms, acronyms, and domain-specific jargon from a meeting transcript or summary.
        For each term, provide a concise and clear definition based on the context in which it appears.
        Focus on terms relevant to the subject matter of the discussion, such as technical terms, industry metrics, product names, or organizational acronyms.
        If a term is ambiguous or used in a non-standard way, include a brief explanation of how it is used in this particular context.
        Exclude generic words or terms that are not specific to the domain of the conversation.
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
        Developed as part of a team communication initiative, I was trained on a massive dataset of meeting transcripts and 
        email best practices to ensure clarity and actionability in all summaries.
        I specialize in synthesizing information from diverse sources into a single, effective communication.
        """
        )
    ),  # This is the backstory of the agent, this helps the agent to understand the context of the task
    goal=dedent(
        (
            """
        Compose an informative, concise, well-structured email with a professional template and tone. Use proper headers, bullet points, section titles and lines. Format the email using HTML for bold headers and line breaks and style it so looks good and professional, avoid inline style. Structure the e-mail as follows:
            1. Greetings (also short overview)
            2. Meeting Summary 
            3. Key Takeaways
            4. Action Items
            5. Insights and Clarifications [MeetingMind Notes]
            6. Glossary
            7. Helpful Links (format the URL and title properly)
            8. Closing line fixed sender name: MeetingMind

        You have access to Email Tool to send the e-mail, Ensure the email body is suitable for direct rendering in a mail client, the content is complete, and parse the result according to the tool requirements before sending.:
            subject (required) - The subject line of the email.
            body (required) - The full content of the email, preferably in HTML format for styling.
            cc (optional) - List of CC recipient email addresses. Example: ["cc1@example.com"]
            bcc (optional) - List of BCC recipient email addresses.
            attachments (optional) - List of file paths or attachment references.

        Only use the tool when you are ready to send, do not send e-mails multiple times
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
        Meeting Summary Coordinator
        """
        )
    ),
    backstory=dedent(
        (
            """
        A seasoned project manager with expertise in AI workflow automation, designed to streamline meeting documentation and communication.
        Possesses deep understanding of natural language processing and information retrieval.
        """
        )
    ),
    goal=dedent(
        (
            """
        Manager to coordinate AI agents to properly and seamlessly, your main objective is to coordinate a complete, accurate, and informative meeting summarization. From transcription retrieval, generating a complete final output, and sending an e-mail. Don't forget to compile and  use outputs from other agents to support the workflow.

        Here are the available agents at your disposal, use accordingly:
            1. Meeting Summary Specialist, Meeting Action Item Extractor, Inquisitive Information Analyst, Meeting Terminology Extractor, each to generate complete and comprehensive output as the building blocks for your final output.
            2. Meeting Email Composer Agent to compile the previous outputs and compose well-structured and pleasant to read e-mail and sending the final output to relevant meeting attendants and stakeholder. Do not send the e-mail multiple times.

        Before invoking the tool, ensure tool_args are normalized into plain strings. If you see {"description": "...", "type": "str"}, replace it with just the value in description.
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
        Given the meeting transcript below:

        {meeting_transcript}

        Please Extract comprehensive meeting information. Leveraging Meeting Summary Specialist, 
        Meeting Action Item Extractor, Inquisitive Information Analyst, Meeting Terminology Extractor agents,
        to generate comprehensive output.
        """
        )
    ),
    expected_output=dedent(
        (
            """
        A comprehensive meeting summary including meeting title, date, and attendance, and many sections compiled from the other agents.

        Pass the output as context to the next task.
        """
        )
    ),
    agent=meeting_orchestration_leader,
)

task2 = Task(
    description=dedent(
        (
            """
        Take the output from previous task, compose a professional meeting e-mail, leveraging Meeting Email Composer Agent. 
        """
        )
    ),
    expected_output=dedent(
        (
            """
        Confirmation satus that the workflow has been completed
        """
        )
    ),
    agent=meeting_orchestration_leader,
)
