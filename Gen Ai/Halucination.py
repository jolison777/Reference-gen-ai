from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

prompt = """
How many casual leaves are provided
to employees in ABC Company?

Answer:
"""

result = generator(
    prompt,
    max_new_tokens=30,
    do_sample=True,
    temperature=0.9
)

print(result[0]["generated_text"])
