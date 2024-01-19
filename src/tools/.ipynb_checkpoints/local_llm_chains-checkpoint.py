## Defines chain structures and response parsing for local models with Transformers

from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline
import re

# Define the parts of the prompt
system = "Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction: You are a movie recommender system."
likes = "The user has liked {likes}, and disliked {dislikes}. "
reviews = "The user has made the following reviews: {history}. "
question = "Would the following be a good recommendation: "
question_probs = "From 0% to 100%, how likely is the following to be a good recommendation: "
target = "{target}?"
no_explain = "STRICTLY respond with 'Yes.' or 'No.'"
no_explain_probs = "STRICTLY respond with a percentage between 0 and 100 "
explain = "STRICTLY respond with: 'Yes.\nExplanation: ' or 'No.\nExplanation: ' followed by rationale for your recommendation, based on characteristics you can infer about the user, in 3 sentences."
explain_probs = "STRICTLY respond with: a percentage between 0% and 100%, followed by rationale for your recommendation, based on characteristics you can infer about the user, in 3 sentences."
end = "### Response: "
        
def get_chain(template: str, model, tokenizer):
    hf = HuggingFacePipeline(pipeline=pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id))
    
    ## Likes / Dislikes chain
    if template == 'base':
        prompt = PromptTemplate.from_template(system + likes + question + target + no_explain + end)
        chain = prompt | hf | process_response
        return chain
    
    # Likes / Dislikes chain, asking for explanation
    if template == 'base_explain':
        prompt = PromptTemplate.from_template(system + likes + question + target + explain + end)
        chain = prompt | hf | process_response
        return chain
    
    # Review history chain
    if template == 'reviews':
        prompt = PromptTemplate.from_template(system + reviews + question + target + no_explain + end)
        chain = prompt | hf | process_response
        return chain
    
    # Review history chain, asking for explanation    
    if template == 'reviews_explain':
        prompt = PromptTemplate.from_template(system + reviews + question + target + explain + end)
        chain = prompt | hf | process_response
        return chain
    
    ## Likes / Dislikes probability chain
    if template == 'base_probs':
        prompt = PromptTemplate.from_template(system + likes + question_probs + target + no_explain_probs + end)
        chain = prompt | hf | process_probs
        return chain
    
    # Likes / Dislikes probability chain, asking for explanation
    if template == 'base_explain_probs':
        prompt = PromptTemplate.from_template(system + likes + question_probs + target + explain_probs + end)
        chain = prompt | hf | process_probs
        return chain
    
    
def process_response(response):
    # Check if the expected delimiter 'Explanation:' is in the response
    if 'Explanation:' in response:
        # Split the response into the 'recommend' and 'explanation' parts
        recommend, explanation = response.split('Explanation:', 1)
        recommend = recommend.strip()
        explanation = explanation.strip()
        
        # Determine the boolean value for the 'recommend' key
        recommend_bool = ('yes' in response.lower()[:10])
        
        # Return the processed dictionary
        return {'recommend': recommend_bool, 'explanation': explanation}
    else:
        res = {'recommend': None, 'explanation': None}
        if len(response) > 10:
            res['explanation'] = response[4:]
        if 'yes' in response[:10].lower():
            res["recommend"] = True
            return res
        elif 'no' in response[:10].lower():
            res["recommend"] = False
            return res
        else:
            return 'ERROR: ' + response
        # Raise an appropriate exception if the delimiter is not found
        # raise ValueError("The response does not contain the expected 'Explanation:' delimiter.")
        
def process_probs(response):
    print(response)
    # Initialize explanation as None (default when there is no explanation)
    explanation = None
    
    # Check if the string starts with a number
    if re.match(r'^\d+', response):
        # Extract the leading number
        number_match = re.match(r'^\d+', response)
        # Convert the extracted number to int
        probability = int(number_match.group(0)) / 100
        
        # Check if the response length is greater than 10 and has an explanation after the number
        if len(response) > 10:
            # Set explanation to the response after the leading number
            explanation = response[len(number_match.group(0)) + 2:].strip()

        # Return a dictionary containing the probability and explanation (if any)
        return {'recommend': probability, 'explanation': explanation}
    else:
        return 'ERROR: ' + response
