from modules.llama_model import generate_llm_response

def summarize_text(content: str, mode: str = "brief") -> str:
    """Summarize text using the local LLM.
    
    Truncates content to fit within the model's context window
    (approximately 2500 characters) to avoid token overflow.
    """
    if mode == "brief":
        instruction = "Summarize the following text in 5 bullet points for quick review."
    else:
        instruction = "Give a detailed summary of the following text with headings and subpoints."

    # Truncate content to fit within context window (~2500 chars ≈ 625 tokens)
    # leaving room for instruction, prompt structure, and output
    max_content_chars = 2500
    if len(content) > max_content_chars:
        content = content[:max_content_chars] + "\n[... content truncated ...]"

    prompt = f"""### Instruction: {instruction}\n\nText:\n{content}\n\n### Summary:"""
    return generate_llm_response(prompt)