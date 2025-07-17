import json
import re
import streamlit as st 
from llm_config import client, system_prompt, tools, available_functions 

def strip_model_thoughts(text: str) -> str:
    """
    Removes common LLM 'thought' patterns, conversational filler, and LaTeX formatting from the response.
    """
    if not isinstance(text, str):
        return str(text) # Ensure it's a string

    # Remove content within various thought tags (case-insensitive)
    text = re.sub(r'<thought>(.*?)</thought>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<thinking>(.*?)</thinking>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<think>(.*?)</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove tool code markers (might appear if direct response)
    text = re.sub(r'<call:tool_code>.*?</call:tool_code>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<tool_code>.*?</tool_code>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove any remaining standalone tags that might be incomplete or misplaced
    text = text.replace("<thought>", "").replace("</thought>", "")
    text = text.replace("<thinking>", "").replace("</thinking>", "")
    text = text.replace("<think>", "").replace("</think>", "")
    text = text.replace("<call:tool_code>", "").replace("</call:tool_code>", "")
    text = text.replace("<tool_code>", "").replace("</tool_code>", "")
    text = re.sub(r'\\boxed{(.*?)}', r'\1', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'\$\$(.*?)\$\$', r'\1', text, flags=re.DOTALL) 
    text = re.sub(r'^(Okay, |Alright, |Sure, |Here is the answer: |The answer is: |Based on the data, |According to the data, )', '', text, flags=re.IGNORECASE).strip()
    text = re.sub(r'\n\s*\n', '\n', text).strip()
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text, flags=re.DOTALL)

    return text

def run_conversation(user_query: str) -> str:
    """
    The main function to handle the conversation with the LLM using Fireworks.
    It orchestrates sending messages, handling tool calls, and getting the final response.
    """
    model_name = "accounts/fireworks/models/qwen3-30b-a3b"
    
    # initialize messages with the system prompt
    messages = [{"role": "system", "content": system_prompt}]

    if 'conversation_history' in st.session_state:
        for past_user_query, past_assistant_response in st.session_state.conversation_history:
            if past_assistant_response != "...":
                messages.append({"role": "user", "content": past_user_query})
                messages.append({"role": "assistant", "content": past_assistant_response})

    messages.append({"role": "user", "content": user_query})

    try:
        # send the conversation and available tools to the model
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto", # the model decides whether to call a function
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        #  check if the model wants to call a function
        if not tool_calls:
            final_response_content = strip_model_thoughts(response_message.content)
            return final_response_content

        messages.append(response_message)

        # execute the function and get the result
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions.get(function_name)
            
            if not function_to_call:
                return f"I encountered an issue: The function '{function_name}' is not recognized."

            try:
                function_args = json.loads(tool_call.function.arguments)
                function_response_data = function_to_call(**function_args)
            except json.JSONDecodeError:
                return "I had trouble understanding the data structure needed for the tool. Could you rephrase your question?"
            except TypeError as e:
                return f"There was a type error when calling the function: {e}. Please check the input values."
            except Exception as e:
                 return f"An error occurred while processing your request with the data tool: {e}"

            # append the function's response to the message history
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": json.dumps(function_response_data),
                }
            )

        # send the updated messages back to the model
        second_response = client.chat.completions.create(
            model=model_name,
            messages=messages, # now includes the system prompt
        )

        response_content = second_response.choices[0].message.content
        final_response_content = strip_model_thoughts(response_content)
        
        return final_response_content

    except Exception as e:
        return "Sorry, I encountered an error while trying to connect to the AI service. Please try again later."

# function to get only the tool call arguments for evaluation
def get_tool_call_arguments(user_query: str):
    """
    Sends a query to the LLM and returns the arguments of the first tool call, if any.
    This is for evaluation purposes, to inspect the tool call before execution.
    """
    model_name = "accounts/fireworks/models/qwen3-30b-a3b"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto", # the model decides whether to call a function
        )
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        if tool_calls:
            tool_call_info = {
                "name": tool_calls[0].function.name,
                "args": json.loads(tool_calls[0].function.arguments)
            }
            return tool_call_info
        else:
            return None
    except Exception as e:
        return None