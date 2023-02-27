"""Make a request to OpenAI API with retrieving the log probs, example"""
import openai
from openai.openai_object import OpenAIObject

prompt = "The quick brown fox jumps over the lazy dog"
"""Make a request using OpenAI client"""
response = openai.Completion.create(
    engine="davinci",
    prompt=prompt,
    max_tokens=50,
    temperature=0.7,
    top_p=0.9,
    n=1,
    stream=False,
    logprobs=1,
    stop=["\n"],
)
#print(response)
probability: OpenAIObject = response.choices[0]['logprobs']['top_logprobs'][0]
print(probability.items())

