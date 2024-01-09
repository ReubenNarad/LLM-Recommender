## Defines the various chains to try. This is a work in progress.

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

system = "You are a reccomender system."

likes = "The user has liked {likes}, and disliked {dislikes}. "

reviews = "The user has made the following reviews: {history}. "

question =  "Will the user like the following movie? "

keyphrases = "Produce a list of phrases may describe good reccomendataions for this user. Additionally, indicate if the target movie would be a good reccomendation:"

# TODO: descriptions are too long! Reduce or remove.
# target = "\nTitle: {target}\nDescription: {target_description}"
target = "\nTitle: {target}"

explain = "\n Additionally, explain your reasoning."

def get_chain(template: str, model):
    
    ## Base Chain
    if template == 'base':
        prompt = ChatPromptTemplate.from_template(system + likes + question + target)
        model = model
        functions = [
            {
                "name": "reccomendation",
                "description": "A reccomender system",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reccomend": {
                            "type": "boolean",
                            "description": "Whether the user will like the target"
                        },
                    },
                    "required": ["reccomend"],
                },
            }
        ]
        chain = prompt | model.bind(function_call={"name": "reccomendation"}, functions=functions)
        return chain
    
    # Description Chain
    if template == 'base_explain':
        prompt = ChatPromptTemplate.from_template(system + likes + question + target + explain)
        model = model

        functions = [
            {
                "name": "reccomendation",
                "description": "A reccomender system",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reccomend": {
                            "type": "boolean",
                            "description": "Whether the user will like the target"
                        },
                        "explanation": {
                            "type": "string",
                            "description": "Step-by-step rationale for your reccomendation, based on characteristics you can infer about the user. 3 sentences",
                        },
                    },
                    "required": ["reccomend", "explanation"],
                },
            }
        ]
        chain = prompt | model.bind(function_call={"name": "reccomendation"}, functions=functions)
        return chain
    
    if template == 'reviews':
        prompt = ChatPromptTemplate.from_template(system + reviews + question + target)
        model = model

        functions = [
            {
                "name": "reccomendation",
                "description": "A reccomender system",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reccomend": {
                            "type": "boolean",
                            "description": "Whether the user will like the target"
                        },
                    },
                    "required": ["reccomend"]
                },
            }
        ]
        chain = prompt | model.bind(function_call={"name": "reccomendation"}, functions=functions)
        return chain
    
    if template == 'reviews_explain':
        prompt = ChatPromptTemplate.from_template(system + reviews + question + target + explain)
        
        functions = [
            {
                "name": "reccomendation",
                "description": "A reccomender system",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reccomend": {
                            "type": "boolean",
                            "description": "Whether the user will like the target"
                        },
                        "explanation": {
                            "type": "string",
                            "description": "Step-by-step rationale for your reccomendation, based on characteristics you can infer about the user. 3 sentences",
                        },
                    },
                    "required": ["reccomend", "explanation"],
                },
            }
        ]
        chain = prompt | model.bind(function_call={"name": "reccomendation"}, functions=functions)
        return chain
    
    if template == 'keyphrases':
        prompt = ChatPromptTemplate.from_template(system + reviews + keyphrases + target)
        
        functions = [
    {
        "name": "reccomendation",
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
                    "description": "Array of keyphrases to describe a good reccomendation for the user ('prefers ___ over ___, 'focuses on movies directed by ___', 'nostalgic for the __s,', dislikes ___, appreciates (plot point type), does (or does not) explore genres, etc..) The array should have exactly 5 items."
            },
            "reccomend": {
                            "type": "boolean",
                            "description": "Whether the user will like the target"
                        }
            },
            "required": ["keyphrases", "reccomend"],
        }
    }
        ]

        chain = prompt | model.bind(function_call={"name": "reccomendation"}, functions=functions)
        return chain
    
    if template == 'keyphrases_explain':
        prompt = ChatPromptTemplate.from_template(system + reviews + keyphrases + target + explain)
        
        functions = [
    {
        "name": "reccomendation",
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
                    "description": "Array of keyphrases to describe a good reccomendation for the user ('prefers ___ over ___, 'focuses on movies directed by ___', 'nostalgic for the __s,', dislikes ___, appreciates (plot point type), does (or does not) explore genres, etc..) The array should have exactly 5 items."
            },
            "reccomend": {
                            "type": "boolean",
                            "description": "Whether the user will like the target"
                        },
            "explanation": {
                            "type": "string",
                            "description": "Step-by-step rationale for your reccomendation, based on characteristics you can infer about the user. 3 sentences",
                        },
            },
            "required": ["keyphrases", "reccomend", "explanation"],
        }
    }
        ]

        chain = prompt | model.bind(function_call={"name": "reccomendation"}, functions=functions)
        return chain