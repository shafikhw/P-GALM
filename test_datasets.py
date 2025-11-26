from datasets import load_dataset
print("Starting load...")
try:
    ds = load_dataset("derek-thomas/ScienceQA", split="validation", streaming=True)
    print("Loaded streaming dataset")
    print(next(iter(ds)))
except Exception as e:
    print(f"Error: {e}")
