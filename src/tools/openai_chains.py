## Defines chain structures, using OpenAI function calling

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Prompt blocks

system = "You are a recommender system."

likes = "The user has liked {likes}, and disliked {dislikes}. "

reviews = "The user has made the following reviews: {history}. "

question =  "Will the user like the following movie? "

keyphrases = "Produce a list of phrases may describe good recommendataions for this user. Additionally, indicate if the target movie would be a good recommendation:"

# TODO: descriptions are too long! Reduce or remove.
# target = "\nTitle: {target}\nDescription: {target_description}"
target = "\nTitle: {target}"

explain = "\n Additionally, explain your reasoning."

def get_chain(template: str, model):
    
    ## Likes / Dislikes chain
    if template == 'base':
        prompt = ChatPromptTemplate.from_template(system + likes + question + target)
        model = model
        functions = [
            {
                "name": "recommendation",
                "description": "A recommender system",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "recommend": {
                            "type": "boolean",
                            "description": "Whether the user will like the target"
                        },
                    },
                    "required": ["recommend"],
                },
            }
        ]
        chain = prompt | model.bind(function_call={"name": "recommendation"}, functions=functions)
        return chain
    
    # Likes / Dislikes chain, asking for explanation
    if template == 'base_explain':
        prompt = ChatPromptTemplate.from_template(system + likes + question + target + explain)
        model = model

        functions = [
            {
                "name": "recommendation",
                "description": "A recommender system",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "recommend": {
                            "type": "boolean",
                            "description": "Whether the user will like the target"
                        },
                        "explanation": {
                            "type": "string",
                            "description": "Step-by-step rationale for your recommendation, based on characteristics you can infer about the user. 3 sentences",
                        },
                    },
                    "required": ["recommend", "explanation"],
                },
            }
        ]
        chain = prompt | model.bind(function_call={"name": "recommendation"}, functions=functions)
        return chain
    
    # Review history chain
    if template == 'reviews':
        prompt = ChatPromptTemplate.from_template(system + reviews + question + target)
        model = model

        functions = [
            {
                "name": "recommendation",
                "description": "A recommender system",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "recommend": {
                            "type": "boolean",
                            "description": "Whether the user will like the target"
                        },
                    },
                    "required": ["recommend"]
                },
            }
        ]
        chain = prompt | model.bind(function_call={"name": "recommendation"}, functions=functions)
        return chain
    
    # Review history chain, asking for explanation    
    if template == 'reviews_explain':
        prompt = ChatPromptTemplate.from_template(system + reviews + question + target + explain)
        
        functions = [
            {
                "name": "recommendation",
                "description": "A recommender system",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "recommend": {
                            "type": "boolean",
                            "description": "Whether the user will like the target"
                        },
                        "explanation": {
                            "type": "string",
                            "description": "Step-by-step rationale for your recommendation, based on characteristics you can infer about the user. 3 sentences",
                        },
                    },
                    "required": ["recommend", "explanation"],
                },
            }
        ]
        chain = prompt | model.bind(function_call={"name": "recommendation"}, functions=functions)
        return chain
    
    # Keyphrases chain
    if template == 'keyphrases':
        prompt = ChatPromptTemplate.from_template(system + reviews + keyphrases + target)
        
        functions = [
    {
        "name": "recommendation",
        "description": "A recommender system",
        "parameters": {
            "type": "object",
            "properties": {
                "keyphrases": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "minItems": 5,
                    "maxItems": 5,
                    "description": "Array of keyphrases to describe a good recommendation for the user ('prefers ___ over ___, 'focuses on movies directed by ___', 'nostalgic for the __s,', dislikes ___, appreciates (plot point type), does (or does not) explore genres, etc..) The array should have exactly 5 items."
            },
            "recommend": {
                            "type": "boolean",
                            "description": "Whether the user will like the target"
                        }
            },
            "required": ["keyphrases", "recommend"],
        }
    }
        ]

        chain = prompt | model.bind(function_call={"name": "recommendation"}, functions=functions)
        return chain

    # Keyphrases chain
    if template == 'keyphrases_explain':
        prompt = ChatPromptTemplate.from_template(system + reviews + keyphrases + target + explain)
        
        functions = [
    {
        "name": "recommendation",
        "description": "A recommender system",
        "parameters": {
            "type": "object",
            "properties": {
                "keyphrases": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "minItems": 5,
                    "maxItems": 5,
                    "description": "Array of keyphrases to describe a good recommendation for the user ('prefers ___ over ___, 'focuses on movies directed by ___', 'nostalgic for the __s,', dislikes ___, appreciates (plot point type), does (or does not) explore genres, etc..) The array should have exactly 5 items."
            },
            "recommend": {
                            "type": "boolean",
                            "description": "Whether the user will like the target"
                        },
            "explanation": {
                            "type": "string",
                            "description": "Step-by-step rationale for your recommendation, based on characteristics you can infer about the user. 3 sentences",
                        },
            },
            "required": ["keyphrases", "recommend", "explanation"],
        }
    }
        ]

        chain = prompt | model.bind(function_call={"name": "recommendation"}, functions=functions)
        return chain
    
        # Likes / Dislikes chain, asking for explanation
    if template == 'base_explain_probs':
        prompt = ChatPromptTemplate.from_template(system + likes + question + target + explain)
        model = model

        functions = [
            {
                "name": "recommendation",
                "description": "A recommender system",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "recommend": {
                            "type": "int",
                            "description": "Likelihood the user will like the target, as a value from 0 to 100"
                        },
                        "explanation": {
                            "type": "string",
                            "description": "Step-by-step rationale for your recommendation, based on characteristics you can infer about the user. 3 sentences",
                        },
                    },
                    "required": ["recommend", "explanation"],
                },
            }
        ]
        chain = prompt | model.bind(function_call={"name": "recommendation"}, functions=functions)
        return chain