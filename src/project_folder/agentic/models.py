"""Pydantic models for all LLM outputs in the agentic module.

Every LLM response that gets parsed is validated against a model here,
so downstream code always deals with well-typed structures instead of
raw dicts. Models use ``extra="ignore"`` to tolerate the occasional
extra key from an LLM without failing validation.
"""

from __future__ import annotations

import json
import re

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from . import config


class SafetyVerdict(BaseModel):
    """Output from the LLM safety evaluator."""

    model_config = ConfigDict(extra="ignore")

    ok: bool = Field(description="Whether the routine is safe to proceed")
    issues: list[str] = Field(
        default_factory=list,
        description="Human-readable safety issues found",
    )
    flagged_ingredients: list[str] = Field(
        default_factory=list,
        description="Exact ingredient names from the input that are unsafe",
    )


class SkinConcern(BaseModel):
    """One skin-science consideration from the research sub-agent."""

    model_config = ConfigDict(extra="ignore")

    concern: str
    evidence: str
    source: str


class IngredientSuggestion(BaseModel):
    """One cosmetic ingredient suggestion from the research sub-agent."""

    model_config = ConfigDict(extra="ignore")

    ingredient: str
    addresses: str
    why: str


class RoutineProduct(BaseModel):
    """One product in the final skincare routine."""

    model_config = ConfigDict(extra="ignore")

    product_name: str
    url: str = ""
    price: str = Field(default_factory=lambda: config.get_section("models").get("price_unavailable_placeholder", "price unavailable"))
    ingredient: str = ""
    reasoning: str = ""
    image_url: str = ""


class RoutineOutput(BaseModel):
    """Final routine JSON emitted by the planner agent."""

    model_config = ConfigDict(extra="ignore")

    routine: list[RoutineProduct] = Field(default_factory=list)
    concerns_addressed: list[str] = Field(default_factory=list)
    disclaimer: str = ""


def _strip_json_fences(text: str) -> str:
    """Remove surrounding ```json fences from LLM output."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    return text.strip()


def parse_and_validate(model_cls: type[BaseModel], content: str) -> BaseModel:
    """Parse JSON from LLM content and validate against a Pydantic model.

    Args:
        model_cls: The Pydantic model class to validate against.
        content: Raw LLM output string (may contain markdown fences).

    Returns:
        Validated model instance.

    Raises:
        json.JSONDecodeError: If no valid JSON found.
        ValidationError: If JSON doesn't match the model schema.
    """
    text = _strip_json_fences(content)
    obj = json.loads(text)
    return model_cls.model_validate(obj)


def parse_list_and_validate(item_cls: type[BaseModel], content: str) -> list[BaseModel]:
    """Parse a JSON array from LLM content and validate each item.

    Args:
        item_cls: Pydantic model class for each array element.
        content: Raw LLM output string (may contain markdown fences).

    Returns:
        List of validated model instances. Empty list if input is empty
        or if any item fails validation.

    Raises:
        json.JSONDecodeError: If no valid JSON found.
        ValidationError: If any item doesn't match the schema.
    """
    text = _strip_json_fences(content)
    raw = json.loads(text)
    if not isinstance(raw, list):
        raise ValidationError.from_exception_data(
            title=item_cls.__name__,
            line_errors=[{"type": "value_error", "msg": "expected a JSON array"}],
        )
    return [item_cls.model_validate(item) for item in raw]


def parse_with_retry(
    model_cls: type[BaseModel],
    messages: list,
    llm,
    max_retries: int = 1,
) -> BaseModel:
    """Call an LLM, parse JSON, validate against a Pydantic model.

    On validation failure, appends the error to the conversation and retries.

    Args:
        model_cls: Pydantic model class to validate against.
        messages: Conversation messages (system + user).
        llm: Chat model to invoke.
        max_retries: Number of retries on validation failure.

    Returns:
        Validated model instance.

    Raises:
        RuntimeError: If validation fails after all retries.
    """
    for attempt in range(max_retries + 1):
        ai: AIMessage = llm.invoke(messages)
        try:
            return parse_and_validate(model_cls, ai.content or "")
        except (json.JSONDecodeError, ValidationError) as e:
            if attempt < max_retries:
                messages = list(messages)
                messages.append(AIMessage(content=ai.content or ""))
                messages.append(
                    HumanMessage(
                        content=(
                            f"Your previous response was invalid: {e}. "
                            f"Respond STRICTLY in valid JSON matching the "
                            f"{model_cls.__name__} schema."
                        )
                    )
                )
            else:
                raise RuntimeError(
                    f"LLM output failed {model_cls.__name__} validation after "
                    f"{max_retries + 1} attempts: {e}"
                ) from e
