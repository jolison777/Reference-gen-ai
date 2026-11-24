from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

prompt = "Bank customer complaint resolution process is"

def run_generation(temp):
    result = generator(
        prompt,
        temperature=temp
    )
    
    print(f"\n========== Temperature = {temp} ==========")
    for i, r in enumerate(result):
        print(f"\nOutput {i+1}:\n{r['generated_text']}")

# test multiple temperatures
run_generation(0.2)
run_generation(0.7)
run_generation(1.0)