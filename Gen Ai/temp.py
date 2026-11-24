
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

prompt = "Artificial Intelligence will"

# Low temperature
result_low = generator(
    prompt,
    max_new_tokens=50,
    temperature=0.2,
    repetition_penalty=1.2,
    no_repeat_ngram_size=3
)

# High temperature
result_high = generator(
    prompt,
    max_new_tokens=50,
    do_sample=True,
    temperature=1.0,
    repetition_penalty=1.2,
    no_repeat_ngram_size=3
)

# Very high temperature
result_very_high = generator(
    prompt,
    max_new_tokens=50,
    do_sample=True,
    temperature=5.0,
    repetition_penalty=1.2,
    no_repeat_ngram_size=3
)

print("Low Temperature Output:\n", result_low[0]['generated_text'])

print("\nHigh Temperature Output:\n", result_high[0]['generated_text'])

print("\nVery High Temperature Output:\n", result_very_high[0]['generated_text'])
