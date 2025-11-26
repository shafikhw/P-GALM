from datasets import load_dataset

ds = load_dataset("derek-thomas/ScienceQA", split="validation")
print("Features:", ds.features)
print("First example keys:", ds[0].keys())
print("First example values (except image):", {k: v for k, v in ds[0].items() if k != 'image'})
