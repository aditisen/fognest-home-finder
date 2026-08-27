"""
GPT-4o judge scoring for RAG evaluation.
Each answer is scored on three dimensions (1–5 scale).
"""
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Load .env so OPENAI_API_KEY is available to ChatOpenAI
from src.config import settings


class EvalScore(BaseModel):
    relevance: int = Field(description="1-5: Does the answer directly address the question?")
    groundedness: int = Field(description="1-5: Is every claim traceable to a cited listing?")
    helpfulness: int = Field(description="1-5: Would a real SF buyer or agent find this useful?")
    reasoning: str = Field(description="One sentence explaining the scores.")


JUDGE_SYSTEM = """You are an expert evaluator for a San Francisco real estate AI assistant.

Score the assistant's answer on three dimensions, each 1–5:

RELEVANCE (1–5)
  5 = Directly answers the question with specific listings
  3 = Partially answers, or addresses the spirit but misses details
  1 = Off-topic or fails to address the question

GROUNDEDNESS (1–5)
  5 = Every listing claim is backed by a cited MLS number [MLS: ...]
  3 = Some claims are cited, some are vague or unverifiable
  1 = Answer makes claims with no listing citations at all

HELPFULNESS (1–5)
  5 = A real buyer or agent would immediately act on this response
  3 = Useful but lacks detail or actionable next steps
  1 = Unhelpful — vague, generic, or no listings surfaced

Return JSON only."""

JUDGE_HUMAN = """Question: {question}

Assistant answer:
{answer}

Score this answer."""


def score_answer(question: str, answer: str) -> EvalScore:
    llm = ChatOpenAI(model="gpt-4o", temperature=0.0, api_key=settings.openai_api_key)
    result = llm.with_structured_output(EvalScore).invoke([
        SystemMessage(content=JUDGE_SYSTEM),
        HumanMessage(content=JUDGE_HUMAN.format(question=question, answer=answer)),
    ])
    return result
