"""
Example VQA prompt builder.

This file is referenced from the VQA experiment YAML config via ``prompt_file``.
The framework dynamically loads ``build_prompt()`` and optionally
``build_system_prompt()`` at runtime.

Users can create their own prompt files by following this pattern:
    - Define ``build_prompt(chunk_start, chunk_end, chunk_index, **kwargs) -> str``
    - Optionally define ``build_system_prompt(**kwargs) -> str``
"""


def build_prompt(
    chunk_start: float,
    chunk_end: float,
    chunk_index: int,
    **kwargs,
) -> str:
    """Build the user prompt for a VQA query on a video chunk.

    Args:
        chunk_start: Start time of the chunk in seconds.
        chunk_end: End time of the chunk in seconds.
        chunk_index: Zero-based index of the chunk.
        **kwargs: Additional context (e.g., output_schema, labels).

    Returns:
        A fully formed prompt string.
    """
    output_schema = kwargs.get("output_schema")

    base = (
        f"Watch this video segment carefully (from {chunk_start:.1f}s to {chunk_end:.1f}s). "
        f"Answer the following question:\n\n"
        f"What are the people in this video doing?"
    )

    if output_schema:
        fields = output_schema.get("fields", {})
        fields_desc = ", ".join(f"{k}: {v}" for k, v in fields.items())
        base += (
            f"\n\nProvide your answer in JSON format with these fields: {fields_desc}. "
            f"Respond with ONLY the JSON object, nothing else."
        )

    return base


def build_system_prompt(**kwargs) -> str:
    """Build the system prompt for VLM context.

    Args:
        **kwargs: Additional context.

    Returns:
        System prompt string.
    """
    return "You are an expert video analyst. Answer concisely and precisely."
