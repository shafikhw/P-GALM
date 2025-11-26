import json
import argparse
import sys
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional

try:
    import datasets
except ImportError:
    raise ImportError("The 'datasets' library is required. Please install it using 'pip install datasets'.")


def load_prompt_template(path: str = "prompt_template.json") -> Dict[str, Any]:
    """
    Loads the prompt template JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        The parsed JSON dictionary.
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_template_by_id(template: Dict[str, Any], template_id: str) -> Dict[str, Any]:
    """
    Retrieves a specific template by its ID from the loaded template data.

    Args:
        template: The full template dictionary containing a "templates" list.
        template_id: The ID of the template to find.

    Returns:
        The dictionary corresponding to the requested template.

    Raises:
        ValueError: If the template_id is not found.
    """
    templates_list = template.get("templates", [])
    for t in templates_list:
        if t.get("id") == template_id:
            return t
    raise ValueError(f"Template with id '{template_id}' not found.")


def load_scienceqa(split: str = "train"):
    """
    Loads the ScienceQA dataset from Hugging Face.

    Args:
        split: The dataset split to load (e.g., "train", "validation", "test").

    Returns:
        The loaded dataset split.
    """
    return datasets.load_dataset("derek-thomas/ScienceQA", split=split)


@dataclass
class QuestionMeta:
    scienceqa_id: str
    subject: str
    topic: str
    category: str
    skill: str
    grade: Any


@dataclass
class Observed:
    question_text: str
    options: List[str]
    image_caption_optional: Optional[str]
    text_context_optional: Optional[str]
    lecture_optional: Optional[str]
    retrieved_knowledge_optional: Optional[str]


@dataclass
class VPGMSkeleton:
    template_id: str
    question_meta: Dict[str, Any]
    observed: Dict[str, Any]
    latent_posteriors: Dict[str, Any] = field(default_factory=dict)
    answer_posterior: Dict[str, Any] = field(default_factory=dict)


def build_scienceqa_skeleton(example: Dict[str, Any], template_id: str, template: Dict[str, Any], override_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Builds a vPGM skeleton dictionary for a single ScienceQA example.

    Args:
        example: A single example from the ScienceQA dataset.
        template_id: The template ID to use.
        template: The full template dictionary (not used in logic but required by signature).
        override_id: Optional ID to use if the example has no ID.

    Returns:
        A dictionary representing the vPGM skeleton.
    """
    # Extract question_meta fields
    # scienceqa_id from 'id' or 'qid', prefer 'id'
    sqa_id = example.get("id")
    if sqa_id is None:
        sqa_id = example.get("qid")
    
    if sqa_id is None and override_id is not None:
        sqa_id = override_id
    
    # Handle potential missing ID gracefully, though dataset usually has it
    if sqa_id is None:
        sqa_id = "unknown"

    question_meta = QuestionMeta(
        scienceqa_id=str(sqa_id),
        subject=example.get("subject", ""),
        topic=example.get("topic", ""),
        category=example.get("category", ""),
        skill=example.get("skill", ""),
        grade=example.get("grade", -1)
    )

    # Extract observed fields
    # text_context_optional from 'hint' or 'context'
    hint = example.get("hint")
    context = example.get("context")
    text_context = hint if hint else context
    
    # Ensure options is a list of strings
    options = example.get("choices", [])
    if not isinstance(options, list):
        options = []

    observed = Observed(
        question_text=example.get("question", ""),
        options=options,
        image_caption_optional=None,
        text_context_optional=text_context,
        lecture_optional=example.get("lecture"),
        retrieved_knowledge_optional=None
    )

    skeleton = VPGMSkeleton(
        template_id="scienceqa_vpgm_4latent_generic",
        question_meta=asdict(question_meta),
        observed=asdict(observed),
        latent_posteriors={},
        answer_posterior={}
    )

    return asdict(skeleton)


def build_skeletons_for_split(split: str = "validation", template_id: str = "scienceqa_vpgm_4latent_generic") -> Dict[str, Any]:
    """
    Builds skeletons for all examples in a given ScienceQA split.

    Args:
        split: The dataset split to process.
        template_id: The template ID to use.

    Returns:
        A dictionary containing the template ID and a list of instance skeletons.
    """
    # Load template
    full_template = load_prompt_template()
    # Verify template exists (raises ValueError if not found)
    _ = get_template_by_id(full_template, template_id)

    # Load dataset
    ds = load_scienceqa(split=split)

    instances = []
    for example in ds:
        skeleton = build_scienceqa_skeleton(example, template_id, full_template)
        instances.append(skeleton)

    return {
        "template_id": template_id,
        "instances": instances
    }


def main():
    parser = argparse.ArgumentParser(description="Build vPGM skeletons from ScienceQA dataset.")
    parser.add_argument("--split", type=str, default="validation", help="The dataset split to use (default: validation)")
    args = parser.parse_args()

    try:
        result = build_skeletons_for_split(split=args.split)
        
        count = len(result["instances"])
        print(f"Built {count} instances.")
        
        if count > 0:
            print("Example instance:")
            print(json.dumps(result["instances"][0], indent=2, ensure_ascii=False))
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
