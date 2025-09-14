import diffusers
print("Available models in diffusers:")
models = [x for x in dir(diffusers) if 'Model' in x or 'Pipeline' in x]
for model in sorted(models):
    print(f"  {model}")
