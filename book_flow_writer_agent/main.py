# |---------------------------------------------------------|
# |                                                         |
# |                 Give Feedback / Get Help                |
# | https://github.com/getbindu/Bindu/issues/new/choose    |
# |                                                         |
# |---------------------------------------------------------|
#
#  Thank you users! We ❤️ you! - 🌻

"""book-flow-writer-agent - A Bindu Agent for writing complete books using AI flows."""

import argparse
import asyncio
import json
import os
import re
import sys
import traceback
from pathlib import Path
from textwrap import dedent
from typing import Any

from bindu.penguin.bindufy import bindufy
from crewai import LLM, Agent, Crew, Process, Task
from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()

# Global variables
outline_crew: Crew | None = None
chapter_crew: Crew | None = None
_initialized = False
_init_lock = asyncio.Lock()


# Pydantic models for book structure
class ChapterOutline(BaseModel):
    """Outline for a single chapter."""

    title: str
    description: str


class BookOutline(BaseModel):
    """Complete book outline."""

    chapters: list[ChapterOutline]


class Chapter(BaseModel):
    """Complete chapter with content."""

    title: str
    content: str


class AgentResponse:
    """Response object from agent execution."""

    def __init__(self, run_id: str, status: str, content: str = ""):
        """Initialize agent response."""
        self.run_id = run_id
        self.status = status
        self.content = content


class CrewInitializationError(Exception):
    """Exception raised when crew initialization fails."""

    pass


class CrewExecutionError(Exception):
    """Exception raised when crew execution fails."""

    CREW_NOT_INITIALIZED = "Crew not initialized"


def _raise_api_key_error() -> None:
    """Raise an error when API keys are not configured."""
    error_msg = (
        "No API key provided. Set OPENAI_API_KEY or OPENROUTER_API_KEY environment variable.\n"
        "For OpenRouter: https://openrouter.ai/keys\n"
        "For OpenAI: https://platform.openai.com/api-keys"
    )
    raise CrewInitializationError(error_msg)


def load_config() -> dict[str, Any]:
    """Load agent configuration from project root."""
    possible_paths = [
        Path(__file__).parent.parent / "agent_config.json",
        Path(__file__).parent / "agent_config.json",
        Path.cwd() / "agent_config.json",
    ]

    for config_path in possible_paths:
        if config_path.exists():
            try:
                with open(config_path) as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️  Error reading {config_path}: {type(e).__name__}")
                continue

    print("⚠️  No agent_config.json found, using default configuration")
    return {
        "name": "book-flow-writer-agent",
        "description": "AI book writing agent using orchestrated flows",
        "version": "1.0.0",
        "deployment": {
            "url": "http://127.0.0.1:3773",
            "expose": True,
            "protocol_version": "1.0.0",
            "proxy_urls": ["127.0.0.1"],
            "cors_origins": ["*"],
        },
        "environment_variables": [
            {"key": "OPENAI_API_KEY", "description": "OpenAI API key for LLM calls", "required": False},
            {"key": "OPENROUTER_API_KEY", "description": "OpenRouter API key for LLM calls", "required": True},
            {"key": "MEM0_API_KEY", "description": "Mem0 API key for memory operations", "required": False},
        ],
    }


async def initialize_outline_crew() -> None:
    """Initialize the book outline crew with agents and tasks."""
    global outline_crew

    openai_api_key = os.getenv("OPENAI_API_KEY")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    model_name = os.getenv("MODEL_NAME", "openai/gpt-4o")

    llm: Any
    try:
        if openai_api_key and not openrouter_api_key:
            llm = LLM(model="gpt-4o", api_key=openai_api_key, temperature=0.7)
            print("✅ Using OpenAI GPT-4o directly")

        elif openrouter_api_key:
            llm = LLM(
                model=model_name,
                api_key=openrouter_api_key,
                base_url="https://openrouter.ai/api/v1",
                temperature=0.7,
            )
            print(f"✅ Using OpenRouter via CrewAI LLM: {model_name}")

            if not os.getenv("OPENAI_API_KEY"):
                os.environ["OPENAI_API_KEY"] = openrouter_api_key

        else:
            _raise_api_key_error()

    except Exception as e:
        print(f"❌ LLM initialization error: {e}")
        print("⚠️ Using mock LLM for testing only")

        class MockLLM:
            def __call__(self, *args: Any, **kwargs: Any) -> str:
                return "Mock response for testing"

        llm = MockLLM()

    # Define Agents for book outline workflow
    researcher_agent = Agent(
        role="Book Researcher",
        goal="Research and gather comprehensive information about the book topic to create a well-informed outline",
        backstory=dedent(
            """
            You are an expert researcher with deep knowledge across various domains.
            You excel at understanding complex topics, identifying key themes, and
            structuring information in a logical, engaging manner. You know how to
            break down broad topics into digestible chapters that flow naturally.
            """
        ),
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )

    outliner_agent = Agent(
        role="Book Outliner",
        goal="Create a detailed, structured outline for the book based on research and the book's goals",
        backstory=dedent(
            """
            You are a master at creating book outlines that engage readers and
            provide clear structure. You understand narrative flow, information
            hierarchy, and how to organize content for maximum impact. You create
            outlines that serve as solid foundations for compelling books.
            """
        ),
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )

    # Define Tasks for book outline
    research_task = Task(
        description=dedent(
            """
            Research the following book topic thoroughly: {topic}

            Book Goal: {goal}

            Your research should include:
            1. Key themes and concepts related to the topic
            2. Current trends and developments in the field
            3. Important subtopics that should be covered
            4. Logical progression of ideas from basic to advanced
            5. Potential areas of interest for readers

            Provide a comprehensive research report that will inform the book outline.
            """
        ),
        expected_output="A detailed research report covering key themes, trends, subtopics, and logical progression for the book.",
        agent=researcher_agent,
    )

    outline_task = Task(
        description=dedent(
            """
            Based on the research report and the book goal, create a comprehensive book outline.

            Topic: {topic}
            Goal: {goal}

            Create an outline with:
            1. 5-10 chapters that logically flow from one to another
            2. Each chapter should have a clear, descriptive title
            3. Each chapter should have a detailed description of what it will cover
            4. The outline should align with the book's goal
            5. Ensure coverage is comprehensive yet focused

            Return the outline as a structured list of chapters with titles and descriptions.
            """
        ),
        expected_output="A structured book outline with 5-10 chapters, each with a title and detailed description.",
        agent=outliner_agent,
        context=[research_task],
    )

    # Create the outline crew
    outline_crew = Crew(
        agents=[researcher_agent, outliner_agent],
        tasks=[research_task, outline_task],
        verbose=True,
        process=Process.sequential,
        memory=False,
    )

    print("✅ Book Outline Crew initialized (2 agents: Researcher, Outliner)")


async def initialize_chapter_crew() -> None:
    """Initialize the chapter writing crew with agents and tasks."""
    global chapter_crew

    openai_api_key = os.getenv("OPENAI_API_KEY")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    model_name = os.getenv("MODEL_NAME", "openai/gpt-4o")

    llm: Any
    try:
        if openai_api_key and not openrouter_api_key:
            llm = LLM(model="gpt-4o", api_key=openai_api_key, temperature=0.7)
        elif openrouter_api_key:
            llm = LLM(
                model=model_name,
                api_key=openrouter_api_key,
                base_url="https://openrouter.ai/api/v1",
                temperature=0.7,
            )
            if not os.getenv("OPENAI_API_KEY"):
                os.environ["OPENAI_API_KEY"] = openrouter_api_key
        else:
            _raise_api_key_error()
    except Exception as e:
        print(f"❌ LLM initialization error: {e}")

        class MockLLM:
            def __call__(self, *args: Any, **kwargs: Any) -> str:
                return "Mock response for testing"

        llm = MockLLM()

    # Define Agents for chapter writing
    writer_agent = Agent(
        role="Chapter Writer",
        goal="Write engaging, informative, and well-structured chapter content",
        backstory=dedent(
            """
            You are an expert writer with a talent for making complex topics
            accessible and engaging. You write in a clear, compelling style that
            keeps readers interested while delivering valuable information. You
            understand how to structure content for maximum readability and impact.
            """
        ),
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )

    editor_agent = Agent(
        role="Content Editor",
        goal="Review and refine chapter content for quality, clarity, and coherence",
        backstory=dedent(
            """
            You are a meticulous editor with a keen eye for detail and flow.
            You ensure that content is polished, professionally written, and
            aligns with the book's overall goals. You improve clarity, fix
            inconsistencies, and enhance readability.
            """
        ),
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )

    # Define Tasks for chapter writing
    write_chapter_task = Task(
        description=dedent(
            """
            Write a comprehensive chapter based on the following information:

            Chapter Title: {chapter_title}
            Chapter Description: {chapter_description}
            Book Topic: {topic}
            Book Goal: {goal}

            Your chapter should:
            1. Start with an engaging introduction
            2. Cover all points mentioned in the chapter description
            3. Include relevant examples and explanations
            4. Maintain a consistent, engaging tone
            5. Be approximately 2000-3000 words
            6. Flow naturally and maintain reader interest
            7. End with a conclusion or transition to the next chapter

            Write in markdown format.
            """
        ),
        expected_output="A complete, well-written chapter (2000-3000 words) in markdown format covering all aspects of the chapter description.",
        agent=writer_agent,
    )

    edit_chapter_task = Task(
        description=dedent(
            """
            Review and refine the written chapter for quality and coherence.

            Chapter Title: {chapter_title}
            Book Goal: {goal}

            Ensure:
            1. Content is clear, engaging, and well-structured
            2. Grammar and style are polished and professional
            3. Information flows logically
            4. Content aligns with the book's overall goal
            5. Tone is consistent and appropriate
            6. All key points are covered thoroughly

            Return the final, polished chapter in markdown format.
            """
        ),
        expected_output="A polished, professionally edited chapter in markdown format that is clear, engaging, and aligns with the book's goals.",
        agent=editor_agent,
        context=[write_chapter_task],
    )

    # Create the chapter crew
    chapter_crew = Crew(
        agents=[writer_agent, editor_agent],
        tasks=[write_chapter_task, edit_chapter_task],
        verbose=True,
        process=Process.sequential,
        memory=False,
    )

    print("✅ Chapter Writing Crew initialized (2 agents: Writer, Editor)")


async def initialize_crew() -> None:
    """Initialize both crews (outline and chapter)."""
    await initialize_outline_crew()
    await initialize_chapter_crew()


def _parse_chapter_from_lines(lines: list[str]) -> list[ChapterOutline]:
    """Parse chapter outlines from text lines."""
    chapters = []
    current_title = None
    current_desc = []

    for line in lines:
        line = line.strip()
        if line.startswith("#") or (line and not line.startswith("-") and not current_title):
            if current_title and current_desc:
                chapters.append(ChapterOutline(title=current_title, description=" ".join(current_desc)))
            current_title = line.strip("#").strip()
            current_desc = []
        elif line and current_title:
            current_desc.append(line)

    if current_title and current_desc:
        chapters.append(ChapterOutline(title=current_title, description=" ".join(current_desc)))

    return chapters


def extract_chapters_from_outline(result: Any) -> list[ChapterOutline]:
    """Extract chapter outlines from CrewAI result."""
    # Try to parse as structured output
    if hasattr(result, "pydantic") and result.pydantic and isinstance(result.pydantic, BookOutline):
        return result.pydantic.chapters

    # Try to extract from tasks_output
    if hasattr(result, "tasks_output") and result.tasks_output:
        for output in result.tasks_output:
            output_str = str(output)
            lines = output_str.split("\n")
            chapters = _parse_chapter_from_lines(lines)
            if chapters:
                return chapters

    # Fallback: create a simple outline
    return [ChapterOutline(title=f"Chapter {i + 1}", description=f"Content for chapter {i + 1}") for i in range(5)]


async def run_outline_crew(topic: str, goal: str) -> list[ChapterOutline]:
    """Run the outline crew and return chapter outlines."""
    global outline_crew

    if not outline_crew:
        raise CrewExecutionError(CrewExecutionError.CREW_NOT_INITIALIZED)

    print(f"📚 Generating book outline for topic: {topic[:50]}...")
    try:
        result = outline_crew.kickoff(
            inputs={
                "topic": topic,
                "goal": goal,
            }
        )
    except Exception as e:
        error_msg = f"Outline crew execution failed: {e!s}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        raise

    chapters = extract_chapters_from_outline(result)
    print(f"📊 Generated outline with {len(chapters)} chapters")
    return chapters


async def run_chapter_crew(chapter_outline: ChapterOutline, topic: str, goal: str) -> Chapter:
    """Run the chapter crew for a single chapter."""
    global chapter_crew

    if not chapter_crew:
        raise CrewExecutionError(CrewExecutionError.CREW_NOT_INITIALIZED)

    print(f"✍️  Writing chapter: {chapter_outline.title}")
    try:
        result = chapter_crew.kickoff(
            inputs={
                "chapter_title": chapter_outline.title,
                "chapter_description": chapter_outline.description,
                "topic": topic,
                "goal": goal,
            }
        )
    except Exception as e:
        error_msg = f"Chapter crew execution failed: {e!s}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        # Return a placeholder chapter on error
        return Chapter(
            title=chapter_outline.title, content=f"# {chapter_outline.title}\n\n{chapter_outline.description}"
        )

    # Extract chapter content from result
    result_str = str(result)

    # Clean up the content
    content = result_str.strip()
    if content.startswith("```"):
        content = re.sub(r"^```\w*\n?|\n?```$", "", content, flags=re.MULTILINE)

    return Chapter(title=chapter_outline.title, content=content)


def combine_chapters_to_book(chapters: list[Chapter], title: str) -> str:
    """Combine all chapters into a single book markdown."""
    book_content = f"# {title}\n\n"
    book_content += "---\n\n"

    for i, chapter in enumerate(chapters, 1):
        book_content += f"\n## Chapter {i}: {chapter.title}\n\n"
        book_content += chapter.content
        book_content += "\n\n---\n\n"

    return book_content


async def write_complete_book(topic: str, goal: str, title: str) -> str:
    """Orchestrate the complete book writing flow."""
    global _initialized

    # Ensure initialization
    async with _init_lock:
        if not _initialized:
            await initialize_crew()
            _initialized = True

    print(f"📖 Starting book writing flow: '{title}'")
    print(f"📝 Topic: {topic}")
    print(f"🎯 Goal: {goal[:100]}...")

    # Step 1: Generate outline
    chapter_outlines = await run_outline_crew(topic, goal)

    # Step 2: Write all chapters
    chapters = []
    for outline in chapter_outlines:
        chapter = await run_chapter_crew(outline, topic, goal)
        chapters.append(chapter)

    # Step 3: Combine into complete book
    book_content = combine_chapters_to_book(chapters, title)

    print(f"✅ Book writing complete! Total length: {len(book_content)} characters")
    return book_content


def _extract_user_input(messages: list[dict[str, str]]) -> str:
    """Extract user input from messages."""
    if not isinstance(messages, list):
        return ""
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            return msg.get("content", "").strip()
    return ""


def _parse_book_request(user_input: str) -> dict[str, str]:
    """Parse user input to extract book details."""
    # Simple parser - you can make this more sophisticated
    lines = user_input.split("\n")

    result = {
        "title": "Untitled Book",
        "topic": user_input[:200],  # Default to first 200 chars
        "goal": "Create an informative and engaging book on the specified topic.",
    }

    for line in lines:
        line = line.strip()
        if line.lower().startswith("title:"):
            result["title"] = line.split(":", 1)[1].strip()
        elif line.lower().startswith("topic:"):
            result["topic"] = line.split(":", 1)[1].strip()
        elif line.lower().startswith("goal:"):
            result["goal"] = line.split(":", 1)[1].strip()

    return result


def _get_usage_instructions() -> str:
    """Get usage instructions for the handler."""
    return (
        "Please provide book details in the following format:\n\n"
        "Title: [Book title]\n"
        "Topic: [Main topic or subject of the book]\n"
        "Goal: [What the book should accomplish]\n\n"
        "Example:\n"
        "Title: The Future of Artificial Intelligence\n"
        "Topic: AI trends, applications, and future developments in 2026\n"
        "Goal: Provide readers with a comprehensive understanding of current AI trends "
        "and prepare them for future innovations in the field."
    )


async def run_agent(messages: list[dict[str, str]]) -> AgentResponse:
    """Run the agent with the given messages and return a response object."""
    # Extract user input
    user_input = _extract_user_input(messages)
    if not user_input:
        return AgentResponse(
            run_id=f"run-{id(messages)}",
            status="COMPLETED",
            content=_get_usage_instructions(),
        )

    # Parse book request
    book_request = _parse_book_request(user_input)

    # Generate complete book
    try:
        book_content = await write_complete_book(
            topic=book_request["topic"], goal=book_request["goal"], title=book_request["title"]
        )

        if book_content and len(book_content) > 500:
            return AgentResponse(
                run_id=f"run-{id(messages)}",
                status="COMPLETED",
                content=book_content,
            )
        else:
            return AgentResponse(
                run_id=f"run-{id(messages)}",
                status="COMPLETED",
                content="I couldn't generate a complete book. Please try providing more detailed information.",
            )
    except Exception as e:
        error_msg = f"Handler error: {e!s}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        return AgentResponse(
            run_id=f"run-{id(messages)}",
            status="ERROR",
            content=f"## Error\n\n{error_msg}\n\nPlease check your API keys and try again.",
        )


async def handler(messages: list[dict[str, str]]) -> str | AgentResponse:
    """Handle incoming agent messages."""
    # Use run_agent for processing
    response = await run_agent(messages)
    return response


async def cleanup() -> None:
    """Clean up resources."""
    global outline_crew, chapter_crew
    print("🧹 Cleaning up Book Flow Writer resources...")
    if outline_crew:
        outline_crew = None
    if chapter_crew:
        chapter_crew = None
    print("✅ Cleanup complete")


def main():
    """Run the main entry point for the Book Flow Writer Agent."""
    parser = argparse.ArgumentParser(description="Bindu Book Flow Writer Agent")
    parser.add_argument(
        "--openrouter-api-key",
        type=str,
        default=os.getenv("OPENROUTER_API_KEY"),
        help="OpenRouter API key (env: OPENROUTER_API_KEY)",
    )
    parser.add_argument(
        "--mem0-api-key",
        type=str,
        default=os.getenv("MEM0_API_KEY"),
        help="Mem0 API key (env: MEM0_API_KEY)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("MODEL_NAME", "openai/gpt-4o-mini"),
        help="Model ID for OpenRouter (env: MODEL_NAME)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to agent_config.json (optional)",
    )
    args = parser.parse_args()

    # Set environment variables from CLI
    if args.openrouter_api_key:
        os.environ["OPENROUTER_API_KEY"] = args.openrouter_api_key
        if not os.getenv("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = args.openrouter_api_key
    if args.mem0_api_key:
        os.environ["MEM0_API_KEY"] = args.mem0_api_key
    if args.model:
        os.environ["MODEL_NAME"] = args.model

    print("📚 Book Flow Writer Agent - AI-powered book creation")
    print("✍️  Capabilities: Book outlining, chapter writing, content editing, complete book generation")
    print("⚙️ Process: Multi-crew flow with outline generation and chapter creation")

    # Load configuration
    config = load_config()

    try:
        print("🚀 Starting Bindu Book Flow Writer Agent server...")
        print(f"🌐 Server will run on: {config.get('deployment', {}).get('url', 'http://127.0.0.1:3773')}")
        bindufy(config, handler)
    except KeyboardInterrupt:
        print("\n🛑 Book Flow Writer Agent stopped")
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        asyncio.run(cleanup())


if __name__ == "__main__":
    main()
