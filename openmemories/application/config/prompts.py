"""Prompts and text constants for CCMemoriesMCP Server."""

from typing import List, Tuple

# Scoring prompt for LLM reranking
# Used by LLMReranker to score document relevance to a query
# Note: Uses OpenAI Structured Outputs with json_schema for guaranteed JSON format
SCORING_PROMPT = """You are a relevance scoring system. Score how relevant a document is to a query.

OUTPUT FORMAT:
Return a JSON object with a "score" field containing a float between 0.0 and 1.0.
Example: {{"score": 0.85}}

SCORING SCALE:
- 1.0  = Perfect match - Document directly answers the query
- 0.9  = Very good match - Has the specific information needed
- 0.7  = Good match - Related with partial coverage
- 0.5  = Moderate match - Same domain, different focus
- 0.3  = Weak match - Minimal overlap with query
- 0.0  = No match - Completely unrelated

Query: "{query}"
Document: "{document}"
"""

# Smart replacement detection prompt
# Used by SmartReplacer to determine if a new memory should replace an existing one
# Note: Uses OpenAI Structured Outputs with json_schema for guaranteed JSON format
REPLACEMENT_DETECTION_PROMPT = """You are a memory replacement detection system. Determine if a new memory should replace an existing one.

OUTPUT FORMAT:
Return a JSON object with the following fields:
- "should_replace": boolean (true if new memory replaces old, false otherwise)
- "confidence": float between 0.0 and 1.0 (confidence in the decision)
- "reason": string (brief explanation, max 50 words)

Example: {{"should_replace": true, "confidence": 0.85, "reason": "Same topic with updated preference"}}

REPLACEMENT CRITERIA (should_replace = true):
- Same topic/subject with updated information or stance
- Contradictory statements about the same thing
- New preference replacing old preference (e.g., "I like cats" -> "I don't like cats anymore")
- Updated facts about the same entity
- Corrections or clarifications of previous statements

DO NOT REPLACE IF (should_replace = false):
- Memories are about different topics or entities
- New memory adds information without contradicting old
- Memories are complementary, not contradictory
- Topics are only superficially similar (e.g., both mention "cat" but different contexts)

CONFIDENCE SCALE:
- 1.0  = Definite replacement - clear update/correction of same topic
- 0.9  = Very high confidence - obvious topical overlap with updated stance
- 0.7  = High confidence - same subject with meaningful change
- 0.5  = Moderate - possibly related but unclear if replacement
- 0.3  = Low confidence - different topics with minor overlap
- 0.0  = No replacement - unrelated memories

Existing Memory: "{old_memory}"
New Memory: "{new_memory}"
"""

# Template components for dynamic MCP_INSTRUCTIONS assembly
INSTRUCTIONS_HEADER = """CCMemoriesMCP Server - Project-based memory storage for intelligent AI Agents.

This server provides persistent memory storage with hybrid search (semantic + full-text)
and RRF fusion ranking.

Available Tools:"""

# Canonical ordering for consistent instruction generation
TOOL_ORDER: List[str] = ["add", "get_all", "search", "remove"]


def build_instructions(tool_snippets: List[Tuple[str, str]]) -> str:
    """Build MCP instructions dynamically from tool snippets.

    Assembles the complete MCP_INSTRUCTIONS string by combining the header
    and tool documentation snippets (in canonical order).

    Args:
        tool_snippets: List of (tool_name, snippet) tuples. Each snippet should
            be a formatted string from the tool's get_instruction_snippet() method.

    Returns:
        Complete MCP instructions string with only the provided tools documented.
        If no tools are provided, returns instructions with "(No tools available)".

    Example:
        >>> snippets = [("add", "    • add(messages: list[str])\\n      ...")]
        >>> instructions = build_instructions(snippets)
    """
    if not tool_snippets:
        return f"{INSTRUCTIONS_HEADER}\n    (No tools available)"

    # Sort by predefined order; unknown tools appear at end alphabetically
    def sort_key(item: Tuple[str, str]) -> Tuple[int, str]:
        name = item[0]
        if name in TOOL_ORDER:
            return (TOOL_ORDER.index(name), name)
        return (len(TOOL_ORDER), name)

    sorted_snippets = sorted(tool_snippets, key=sort_key)

    # Assemble the tool section
    tool_section = "\n\n".join(snippet for _, snippet in sorted_snippets)

    return f"{INSTRUCTIONS_HEADER}\n{tool_section}"


# Backward compatibility: static fallback with all tools (for direct imports)
# This generates the same output as the previous static MCP_INSTRUCTIONS
MCP_INSTRUCTIONS = build_instructions(
    [
        (
            "add",
            "    • add(messages: list[str])\n"
            "      Add messages with semantic embeddings. Empty lists are no-op.\n"
            "      Messages must be 1-30720 characters, non-whitespace.",
        ),
        (
            "get_all",
            "    • get_all() -> list[str]\n"
            "      Retrieve all stored messages. Returns empty list if none stored.",
        ),
        (
            "search",
            "    • search(query: str) -> list[str]\n"
            "      Hybrid semantic + full-text search. Finds semantically similar\n"
            "      messages using vector embeddings (limit: configurable, default 5).",
        ),
        (
            "remove",
            "    • remove(messages: list[str])\n"
            "      Remove messages using exact string matching (case-sensitive).\n"
            "      Uses semantic search to find candidates, then exact match filter.\n"
            "      Removes all occurrences of each message. Silently ignores non-existent messages.",
        ),
    ]
)
