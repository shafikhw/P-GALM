import json
from typing import Dict, Any
from scienceqa_vpgm_loader import load_prompt_template, get_template_by_id

def pretty(obj: Any) -> str:
    """
    Returns a pretty-printed JSON string for the given object.
    
    Args:
        obj: The object to serialize.
        
    Returns:
        A formatted JSON string.
    """
    return json.dumps(obj, indent=2, ensure_ascii=False)

def build_vpgm_prompt(
    skeleton: Dict[str, Any],
    template: Dict[str, Any],
    ensure_json_output: bool = True
) -> str:
    """
    Builds the textual prompt for the LLM based on the skeleton and template.

    Args:
        skeleton: The dictionary containing the ScienceQA example data.
        template: The dictionary containing the vPGM template.
        ensure_json_output: Whether to include instructions for JSON output.

    Returns:
        A single string containing the complete prompt.
    """
    observed_str = pretty(skeleton.get("observed", {}))
    meta_str = pretty(skeleton.get("question_meta", {}))
    cpd_templates_str = pretty(template.get("verbal_cpd_templates", {}))
    instance_fields_str = pretty(template.get("instance_fields", {}))

    prompt_parts = [
        "You are performing verbalized probabilistic graphical model inference.",
        "",
        "### Task",
        "",
        "Given the following ScienceQA question, fill in the missing latent_posteriors and answer_posterior according to the template.",
        "",
        "### Observed Data",
        "",
        observed_str,
        "",
        "### Metadata",
        "",
        meta_str,
        "",
        "### Latent Variable Instructions",
        "",
        "For each latent variable, follow these instructions:",
        cpd_templates_str,
        "",
        "### Output Format (MUST match exactly)",
        "",
        "You must output ONLY a JSON object with the following structure:",
        instance_fields_str,
        "",
        "### Your Output",
        "",
        "Fill in:",
        "",
        "* latent_posteriors",
        "* answer_posterior",
        "",
        "Produce only valid JSON."
    ]

    return "\n".join(prompt_parts)

def build_prompt_for_instance(skeleton: Dict[str, Any], template_id: str = "scienceqa_vpgm_4latent_generic") -> str:
    """
    Convenience function to build a prompt for a given skeleton instance.
    Loads the template from disk.

    Args:
        skeleton: The dictionary containing the ScienceQA example data.
        template_id: The ID of the template to use.

    Returns:
        A single string containing the complete prompt.
    """
    template_full = load_prompt_template()
    template = get_template_by_id(template_full, template_id)
    return build_vpgm_prompt(skeleton, template)
