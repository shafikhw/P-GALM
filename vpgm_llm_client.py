import json
import os
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import dotenv
dotenv.load_dotenv()
from openai import OpenAI, APIError

from scienceqa_vpgm_loader import load_prompt_template, get_template_by_id
from build_vpgm_llm_prompt import build_vpgm_prompt, build_prompt_for_instance

def get_openai_client() -> Any:
    """
    Configures and returns an openai API client.
    Reads OPENAI_API_KEY from environment.
    
    Returns:
        OpenAI client instance.
        
    Raises:
        RuntimeError: If OPENAI_API_KEY is not set.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
    
    return OpenAI(api_key=api_key)

def call_llm_with_prompt(
    prompt: str,
    model: str = "gpt-4.1",
    temperature: float = 0.0,
    max_tokens: int = 2048
) -> str:
    """
    Calls the LLM with the given prompt.
    
    Args:
        prompt: The full text prompt.
        model: The model identifier.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        
    Returns:
        The content of the assistant's response.
        
    Raises:
        RuntimeError: If the API call fails.
    """
    client = get_openai_client()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"LLM call failed: {e}")

def extract_json_from_text(raw_output: str) -> str:
    """
    Extracts JSON string from raw text output.
    
    Args:
        raw_output: The raw string from the LLM.
        
    Returns:
        A clean JSON string.
        
    Raises:
        ValueError: If JSON cannot be found or parsed.
    """
    # First try parsing as is
    try:
        json.loads(raw_output)
        return raw_output
    except json.JSONDecodeError:
        pass
    
    # Heuristic extraction
    start_idx = raw_output.find('{')
    end_idx = raw_output.rfind('}')
    
    if start_idx == -1 or end_idx == -1 or start_idx > end_idx:
        raise ValueError("No JSON object found in output.")
        
    candidate = raw_output[start_idx : end_idx + 1]
    
    try:
        json.loads(candidate)
        return candidate
    except json.JSONDecodeError as e:
        raise ValueError(f"Extracted text is not valid JSON: {e}")

def parse_vpgm_instance(raw_output: str) -> Dict[str, Any]:
    """
    Parses the raw LLM output into a dictionary.
    
    Args:
        raw_output: The raw string from the LLM.
        
    Returns:
        The parsed dictionary.
        
    Raises:
        ValueError: If parsing fails.
    """
    json_str = extract_json_from_text(raw_output)
    return json.loads(json_str)

def validate_probability_dict(
    probs: Dict[str, float],
    tolerance: float = 1e-3
) -> None:
    """
    Validates a probability dictionary.
    
    Args:
        probs: Dictionary mapping keys to probabilities.
        tolerance: Tolerance for sum check.
        
    Raises:
        ValueError: If validation fails.
    """
    if not probs:
        raise ValueError("Probability dictionary is empty.")
        
    total = 0.0
    for k, v in probs.items():
        if not isinstance(v, (int, float)):
            raise ValueError(f"Value for '{k}' is not a number: {v}")
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Value for '{k}' is out of range [0, 1]: {v}")
        total += v
        
    if abs(total - 1.0) > tolerance:
        raise ValueError(f"Probabilities sum to {total}, expected 1.0 (tolerance {tolerance}).")

def validate_vpgm_instance_against_template(
    instance: Dict[str, Any],
    template: Dict[str, Any]
) -> None:
    """
    Validates the parsed instance against the template structure.
    
    Args:
        instance: The parsed vPGM instance.
        template: The template definition.
        
    Raises:
        ValueError: If validation fails.
    """
    # Top level keys
    required_keys = ["template_id", "question_meta", "observed", "latent_posteriors", "answer_posterior"]
    for key in required_keys:
        if key not in instance:
            raise ValueError(f"Missing top-level key: {key}")
            
    # Template ID check
    expected_id = template.get("id")
    # Also check inside instance_fields if present, as per instructions
    if not expected_id and "instance_fields" in template:
        expected_id = template["instance_fields"].get("template_id")
        
    if instance["template_id"] != expected_id:
        raise ValueError(f"Template ID mismatch. Expected '{expected_id}', got '{instance['template_id']}'")
        
    # Latent variables
    template_latents = template.get("instance_fields", {}).get("latent_posteriors", {})
    instance_latents = instance["latent_posteriors"]
    
    for latent_name in template_latents:
        if latent_name not in instance_latents:
            raise ValueError(f"Missing latent variable: {latent_name}")
            
        latent_data = instance_latents[latent_name]
        if "state_probabilities" not in latent_data:
            raise ValueError(f"Missing 'state_probabilities' for latent: {latent_name}")
        if "justification" not in latent_data:
            raise ValueError(f"Missing 'justification' for latent: {latent_name}")
            
        validate_probability_dict(latent_data["state_probabilities"])
        
    # Answer posterior
    answer_posterior = instance["answer_posterior"]
    if "option_probabilities" not in answer_posterior:
        raise ValueError("Missing 'option_probabilities' in answer_posterior")
    if "selected_answer" not in answer_posterior:
        raise ValueError("Missing 'selected_answer' in answer_posterior")
        
    validate_probability_dict(answer_posterior["option_probabilities"])
    
    # Selected answer check
    options = instance.get("observed", {}).get("options")
    selected = answer_posterior["selected_answer"]
    if isinstance(options, list) and options:
        if selected not in options:
            raise ValueError(f"Selected answer '{selected}' is not in observed options: {options}")

def infer_vpgm_for_skeleton(
    skeleton: Dict[str, Any],
    template_full: Dict[str, Any],
    template_id: str = "scienceqa_vpgm_4latent_generic",
    model: str = "gpt-4.1",
    max_retries: int = 3,
    retry_sleep_seconds: float = 1.0
) -> Dict[str, Any]:
    """
    Performs full inference for a single skeleton.
    
    Args:
        skeleton: The input skeleton.
        template_full: The full template object.
        template_id: The template ID.
        model: The LLM model to use.
        max_retries: Number of retries on failure.
        retry_sleep_seconds: Sleep time between retries.
        
    Returns:
        The validated vPGM instance.
        
    Raises:
        RuntimeError: If inference fails after all retries.
    """
    template = get_template_by_id(template_full, template_id)
    prompt = build_vpgm_prompt(skeleton, template)
    # print (prompt)
    last_error = None
    
    for attempt in range(max_retries):
        try:
            raw_output = call_llm_with_prompt(prompt, model=model)
            instance = parse_vpgm_instance(raw_output)
            validate_vpgm_instance_against_template(instance, template)
            return instance
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(retry_sleep_seconds)
                
    raise RuntimeError(f"Inference failed after {max_retries} retries. Last error: {last_error}")

def infer_vpgm_for_instances(
    skeletons: List[Dict[str, Any]],
    template_id: str = "scienceqa_vpgm_4latent_generic",
    model: str = "gpt-4.1"
) -> List[Dict[str, Any]]:
    """
    Runs inference for a batch of skeletons.
    
    Args:
        skeletons: List of skeleton dictionaries.
        template_id: Template ID.
        model: LLM model.
        
    Returns:
        List of validated instances.
    """
    template_full = load_prompt_template()
    results = []
    
    for skeleton in skeletons:
        # We let exceptions propagate or could catch them here. 
        # Instructions imply returning a list, but if one fails, 
        # infer_vpgm_for_skeleton raises RuntimeError. 
        # We will let it raise as per typical strict batch processing or 
        # if the user wanted partial results they would have specified error handling here.
        # Given "Does not print anything", we assume strict failure or propagation.
        instance = infer_vpgm_for_skeleton(
            skeleton, 
            template_full, 
            template_id=template_id, 
            model=model
        )
        results.append(instance)
        
    return results

def main():
    try:
        template_full = load_prompt_template()
        
        # Create a dummy skeleton
        dummy_skeleton = {
            "template_id": "scienceqa_vpgm_4latent_generic",
            "question_meta": {
                "scienceqa_id": "dummy_001",
                "subject": "natural science",
                "topic": "biology",
                "category": "animals",
                "skill": "classification",
                "grade": "grade1"
            },
            "observed": {
                "question_text": "Which animal is a mammal?",
                "options": ["Shark", "Dog", "Eagle"],
                "image_caption_optional": None,
                "text_context_optional": None,
                "lecture_optional": None,
                "retrieved_knowledge_optional": None
            },
            "latent_posteriors": {},
            "answer_posterior": {}
        }
        
        print("Running inference on dummy skeleton...")
        # Note: This will fail if OPENAI_API_KEY is not set, which is expected behavior.
        result = infer_vpgm_for_skeleton(
            dummy_skeleton, 
            template_full, 
            template_id="scienceqa_vpgm_4latent_generic"
        )
        
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
    except Exception as e:
        # Print error to stderr but don't crash the script if it's just a demo failure
        import sys
        print(f"Error in main: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
