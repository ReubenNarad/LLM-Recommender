## Defines the various chains to try. This is a work in progress.

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


def define_chain(template: str, model):
    
    ## Base Chain
    if template == 'Base':
        prompt = ChatPromptTemplate.from_template("You are a reccomender system. User liked {likes}, User Disliked {dislikes}. Will the user like {target}?")
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
    
    ## Rationale Chain
    if template == 'Base + Rationale':
        prompt = ChatPromptTemplate.from_template("You are a reccomender system. User liked {likes}, User Disliked {dislikes}. Will the user like {target}?")
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
                            "description": "Detailed rationale for your reccomendation. 3 sentences",
                        },
                    },
                    "required": ["reccomend", "explanation"],
                },
            }
        ]
        chain = prompt | model.bind(function_call={"name": "reccomendation"}, functions=functions)
        return chain
    
    # Description Stuff Chain
    if template == 'Base + Description':
        prompt = ChatPromptTemplate.from_template("You are a reccomender system. User liked {likes}, User Disliked {dislikes}. Will the user like {target}?")
        model = ChatOpenAI(model_name="gpt-3.5-turbo")

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
                            "description": "Step-by-step rationale for your reccomendation, including user demographics. 3 sentences",
                        },
                    },
                    "required": ["reccomend", "explanation"],
                },
            }
        ]
        chain = prompt | model.bind(function_call={"name": "reccomendation"}, functions=functions)
        return chain
    
# chain = define_chain('Base', ChatOpenAI)
# print(chain)
# print(chain.middle)