import json
import logging
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import our existing modules
from scienceqa_vpgm_loader import load_scienceqa, load_prompt_template, build_scienceqa_skeleton, get_template_by_id
from vpgm_llm_client import infer_vpgm_for_skeleton

# Configure logging to print to terminal
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("vpgm_monitor")

app = FastAPI(title="ScienceQA vPGM Explorer")

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
STATE = {
    "dataset": None,
    "template": None,
    "template_id": "scienceqa_vpgm_4latent_generic"
}

@app.on_event("startup")
async def startup_event():
    logger.info("Loading ScienceQA dataset and templates... This may take a moment.")
    # Load validation split by default for exploration
    STATE["dataset"] = load_scienceqa(split="validation")
    STATE["template"] = load_prompt_template()
    logger.info(f"Loaded {len(STATE['dataset'])} examples from ScienceQA validation split.")

def get_example_by_id(sqa_id: str):
    # ID format: "idx_{index}"
    if sqa_id.startswith("idx_"):
        try:
            idx = int(sqa_id.split("_")[1])
            ds = STATE["dataset"]
            if 0 <= idx < len(ds):
                return ds[idx]
        except (ValueError, IndexError):
            pass
            
    # Fallback to searching if we ever have real IDs
    ds = STATE["dataset"]
    for ex in ds:
        if str(ex.get("id")) == sqa_id or str(ex.get("qid")) == sqa_id:
            return ex
    return None

@app.get("/api/questions")
async def list_questions(page: int = 1, limit: int = 20, search: Optional[str] = None):
    ds = STATE["dataset"]
    if not ds:
        raise HTTPException(status_code=503, detail="Dataset not loaded")

    # Filter out images first (as per requirement) and apply search
    filtered_indices = []
    search_lower = search.lower() if search else None
    
    # Iterate to filter
    # Note: For 4k items this is fast enough.
    for i in range(len(ds)):
        ex = ds[i]
        
        # Requirement: Omit questions with images
        if ex.get("image") is not None:
            continue
            
        # Search filter
        if search_lower:
            q_text = ex.get("question", "")
            # Check question text
            if search_lower not in q_text.lower():
                continue
                
        filtered_indices.append(i)
            
    # Pagination
    total_filtered = len(filtered_indices)
    start = (page - 1) * limit
    end = start + limit
    page_indices = filtered_indices[start:end]
    
    results = []
    for i in page_indices:
        ex = ds[i]
        results.append({
            "id": f"idx_{i}",
            "question": ex.get("question"),
            "subject": ex.get("subject"),
            "topic": ex.get("topic")
        })
        
    return {
        "total": total_filtered,
        "page": page,
        "limit": limit,
        "items": results
    }

@app.get("/api/questions/{sqa_id}")
async def get_question_details(sqa_id: str):
    ex = get_example_by_id(sqa_id)
    if not ex:
        raise HTTPException(status_code=404, detail="Question not found")
    
    # Build skeleton to show what the model sees
    skeleton = build_scienceqa_skeleton(ex, STATE["template_id"], STATE["template"], override_id=sqa_id)
    
    # Sanitize example for JSON response
    # 1. Convert to dict (if it's a dataset row)
    # 2. Remove 'image' field (PIL Image object causes serialization error)
    ex_dict = dict(ex)
    if "image" in ex_dict:
        del ex_dict["image"]
    
    return {
        "raw_example": ex_dict,
        "skeleton": skeleton
    }

@app.post("/api/infer/{sqa_id}")
async def run_inference(sqa_id: str):
    ex = get_example_by_id(sqa_id)
    if not ex:
        raise HTTPException(status_code=404, detail="Question not found")
    
    skeleton = build_scienceqa_skeleton(ex, STATE["template_id"], STATE["template"], override_id=sqa_id)
    
    logger.info(f"--- Running Inference for {sqa_id} ---")
    
    try:
        # Run inference
        result = infer_vpgm_for_skeleton(
            skeleton,
            STATE["template"],
            template_id=STATE["template_id"]
        )
        
        # MONITORING: Print full JSON to terminal
        print("\n" + "="*40)
        print(f"FULL JSON RESPONSE FOR {sqa_id}:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print("="*40 + "\n")
        
        return result
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
