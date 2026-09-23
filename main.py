import os
from pathlib import Path

from dotenv import load_dotenv
from openai import(
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
)

# ========== PYTHON: LOAD LOCAL CONFIGURATION ==========

PROJECT_DIRECTORY = Path(__file__).resolve().parent
ENV_FILE = PROJECT_DIRECTORY / ".env"

# Show the expected location if .env is missing.
if not ENV_FILE.is_file():
    raise FileNotFoundError(f"Configuration file not found: {ENV_FILE}")

# Load local settings, including files saved with a UTF-8 BOM.
load_dotenv(ENV_FILE, override=True, encoding="utf-8-sig")
 
 
#Keep a limited number of completed conversations turns in memory.
#This limits message count not tokens; long messages can still
#exceed the model's context window

MAX_HISTORY_TURNS = 6


# describe only the capabilities currently connected to this chatbot

SYSTEM_PROMPT=(
    "YOU are the conversational assistant for a job application project"
    "Help the user discuss job searches, resumes and this project"
    "Currently, you can only converse"
    "job discovery resume files application submisson reporting bugs"
    "and code maintenance tools are not connected yet"
    "Never claim to have performed actions or inspected information"
    "that you cannot access. If you are unsure, respond with 'I don't know' or 'I cannot do that' "
    "Do not invent the user's qualification or experience."
    "keep replies clear and concise"
    
) 


#====Python Message Validation===============
def validate_message(message: str) -> str:
    """ Return cleaned text, or reject an  empty message"""
    # remove whitespace around the message without chaning its wording.
    Cleaned_message = message.strip()
    
    #Prevent empty requests from reaching the model
    if not Cleaned_message:
        raise ValueError("Empty message is not allowed.")
    return Cleaned_message


#====== PYTHON: REQUEST AN LLM RESPONCE===========

def get_llm_response(
    client: OpenAI,
    model: str,
    message: str,
    history: list[dict[str,str]],
) -> str:
    """Send a message to the LLM and return its response."""

    # build a new request without modifying saved conversation history.
    message = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history,
        {"role":"user","content": message},
    ]    

    # Request a reply from the local OpenAI-compatible Qwen server
    response = client.chat.completions.create(
        model=model,
        messages=message,
        max_tokens=512,
    )
    
    #check that the server supplies a response choice 
    if not response.choices:
        raise ValueError("The model returnes no response choice.")
    
       # Extract the answer without assuming text was returned.
    reply = response.choices[0].message.content

    # Reject missing or blank output before calling strip().
    if not reply or not reply.strip():
        raise ValueError("The model returned an empty answer")

    return reply.strip()
    
    ##avoid treating missing or blank output as a succuessful reply.
    if not reply or not reply.strip():
        raise ValueError("The model returnes an empty answer")
    return reply.strip()


#=========== Python: Terminal Chat loop ===========
def run_chat()-> None:
    """ Run the terminal using the configured local model."""
    
    # read connection settings from enviroment variables loaded from .env
    api_key = os.getenv("LLM_API_KEY","").strip()
    base_url = os.getenv("LLM_BASE_URL","").strip()
    model = os.getenv("LLM_MODEL","").strip() 

    # Identify missing settings without printing credentials.
    settings = {
        "LLM_API_KEY": api_key,
        "LLM_BASE_URL": base_url,
        "LLM_MODEL": model,
    }
    
    
    missing_settings = [
        name for name, value in settings.items() if not value
    ]

    #stop early if the connection configuration is incomplete
    if missing_settings:
        print(
            "Configuration error: fill in these settings in .env:"
            + ",".join(missing_settings)
        )
        return
    
    
    
    # store conversation history only for this running session.
    history: list[dict[str,str]] = []
    
    
    print("Job Agent Terminal")
    print(f"Type 'exit' to close or 'clear' to reset conversation memory")
    
    
    #reuse one connection client and close it when the session ends.
    # Local geberation may take time, so allow up to 120 secounds 
    
    with OpenAI( 
                api_key=api_key,
                base_url=base_url,
                timeout=120.0,
                max_retries = 0,
                ) as client:
                    while True:
                        #read input and allow a clean keyboard exit.
                        try:
                            user_message = input("\n You: ")
                            
                        except (KeyboardInterrupt, EOFError):
                            print(f"\n CHAT CLOSED")
                            break
                        
                        # Reject Blank input and ask again
                        try:
                            validated_message = validate_message(user_message)
                            
                        except ValueError as error:
                            print(f"Input error {error}")
                            continue
                        
                        #match terminal controls without changing the original message
                        command = validated_message.lower()
                        
                        if command == "exit":
                            print("chat closed")
                            break
                        
                        if command == "clear":
                            history.clear()
                            print(f"Conversation memory cleard")
                            continue
                        
                        #request a real response from Qwen.
                        try:
                            print(f"Agent: THinking....")
                            
                            reply = get_llm_response(
                                client=client,
                                model=model,
                                message=validated_message,
                                history=history,
                            )
    
                        except APITimeoutError:
                            print(
                                "Requested time out. Check the qwen server terminal"
                                "before trying again"
                            )
                            
                            continue 
                        
                        except APIConnectionError:
                            print(
                                "cannot connect to Qwen. Keep llama-server running"
                                "and check LLM_BASE_URL in .env"
                            )        
                            continue
                        
                        except APIStatusError as error:
                            #Display http statis without exposing request contents
                            print(
                                f"Model server returned HTTP{error.status_code}"
                                f"Check the server terminal for details."
                            )
                            continue 
                        except APIError:
                            print(
                                "the model request failed"
                                "Chcek the Qwen server terminal for details"
                                
                            )
                            
                            continue
                        
                        except ValueError as error:
                            print(f"Response error: {error}")
                            continue
                        
                        except KeyboardInterrupt:
                            # closing the client does not gurantee server generation stops.
                            print("\n CHAT closed ")
                            break
                        
                        #Display and re,ember only successfully completed exchanges.
                        print(f"Agent: {reply}")
                        
                        history.extend([
                            {"role": "user", "content": validated_message},
                            {"role":"assistant","content": reply},
                            
                         ])
                        #Each completed turn contains one user and one assistant message.
                        history = history[-MAX_HISTORY_TURNS*2:]
                            


# ========== PYTHON: APPLICATION ENTRY POINT ==========

# Start the terminal interface only when this file is run directly.
if __name__ == "__main__":
    run_chat()
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            







