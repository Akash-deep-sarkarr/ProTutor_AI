import os
import warnings

# Try to import and load the local Llama model. If the
# native package or the model file is not available, fall
# back to a harmless stub so the Streamlit UI can still load.
llm = None
MODEL_PATH = os.path.join("models", "mistral-7b-instruct-v0.1.Q4_K_M.gguf")

try:
    from llama_cpp import Llama

    try:
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=8192,
            n_threads=8,
            n_gpu_layers=20,
            verbose=False
        )
    except Exception as e:
        llm = None
        warnings.warn(f"Failed to load Llama model from '{MODEL_PATH}': {e}")
except Exception:
    llm = None
    warnings.warn("`llama_cpp` not available — running without local LLM.")


def generate_llm_response(prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
    """Generate a response using the local LLM when available.

    If the LLM is not available, return a helpful placeholder
    message so the UI remains usable for testing.
    """
    if llm is None:
        return (
            "[Local LLM unavailable] Install `llama-cpp-python` and place the GGUF model"
            " file at `models/mistral-7b-instruct-v0.1.Q4_K_M.gguf` to enable full functionality."
        )

    output = llm(prompt=prompt, max_tokens=max_tokens, temperature=temperature)
    # Some versions of llama-cpp-python return different structures; handle defensively
    try:
        return output["choices"][0]["text"].strip()
    except Exception:
        return str(output)