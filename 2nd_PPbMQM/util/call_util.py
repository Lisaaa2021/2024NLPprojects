import replicate
from openai import OpenAI
import json

def call_llama(prompt, min_tokens=0, max_tokens = 512,temperature = 0.6, top_p = 0.9, top_k = 50, frequency_penalty =0.2, presence_penalty= 1.15):
    system_prompt = 'You are a helpful assistant.'
    print(prompt)
    input = {
    "min_tokens" : min_tokens,
    "max_tokens":max_tokens,
    "seed" : 240425,
    "top_p": top_p,
    "top_k": top_k,
    "temperature": temperature,
    "system_prompt": system_prompt,
    "presence_penalty": presence_penalty,
    "frequency_penalty": frequency_penalty,
    "prompt": prompt,
    #"prompt_template":prompt_template,
        }

    for event in replicate.stream(
        "meta/meta-llama-3-70b-instruct",
        input=input):
        print(event, end="")





def call_gpt(prompt_1, model = "gpt-3.5-turbo-0125", seed = 240425, max_tokens = 512,temperature = 0.2, top_p = 1,frequency_penalty =0, presence_penalty= 0):
    client = OpenAI()
    print(prompt_1)
    response = client.chat.completions.create(
    model= model,
    response_format= {"type":"json_object"},
    seed = seed,
    max_tokens=max_tokens,
    temperature = temperature,
    top_p= top_p,
    frequency_penalty = frequency_penalty,
    presence_penalty = presence_penalty,
   # response_format={ "type": "json_object" },
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt_1}
       # {"role": "user","content": prompt_2}
    ])
    result = response.choices[0].message.content
    fingerprint = response.system_fingerprint

    print('response.model_dump_json():')
    print(json.dumps(json.loads(response.model_dump_json()), indent=4))

    return result, fingerprint


def call_gpt_zero_shot(system_prompt,instruction_prompt,cn,en,model = "gpt-3.5-turbo-0125", seed = 240425, max_tokens = 512,temperature = 1, top_p = 1,frequency_penalty =0, presence_penalty= 0):
    client = OpenAI()
    response = client.chat.completions.create(
    model= model,
    seed = seed,
    max_tokens=max_tokens,
    temperature = temperature,
    top_p= top_p,
    frequency_penalty = frequency_penalty,
    presence_penalty = presence_penalty,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instruction_prompt},
        {"role": "assistant","content": "Ok, I am ready. Please provide me with a Chinese source sentence and its English target translation."},
        {"role": "user", "content": f"Source: {cn}; Target: {en}"}
    ])
    result = response.choices[0].message.content
    fingerprint = response.system_fingerprint

    print('response.model_dump_json():')
    print(json.dumps(json.loads(response.model_dump_json()), indent=4))

    return result, fingerprint


def call_llama_zero_shot(instruction_prompt, system_prompt, cn, en, min_tokens=0, max_tokens = 512,temperature = 0.6, top_p = 0.9, top_k = 50, frequency_penalty =0.2, presence_penalty= 1.15):

    prompt_template = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>{system_prompt}<|eot_id|>

<|start_header_id|>user<|end_header_id|>{instruction_prompt}<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>Ok, I am ready. Please provide me with a Chinese source sentence and its English target translation.<|eot_id|>

<|start_header_id|>user<|end_header_id|>Source: {cn}; Target: {en}<|eot_id|>"""

    input = {
    "min_tokens" : min_tokens,
    "max_tokens":max_tokens,
    "seed" : 240425,
    "top_p": top_p,
    "top_k": top_k,
    "temperature": temperature,
    "system_prompt": system_prompt,
    "presence_penalty": presence_penalty,
    "frequency_penalty": frequency_penalty,
    "prompt": instruction_prompt,
    "prompt_template": prompt_template}

    for event in replicate.stream(
        "meta/meta-llama-3-70b-instruct",
        input=input):
        print(event, end="")
