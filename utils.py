from litellm import completion
from typing import List, Optional, Dict
import time
import litellm
#litellm.set_verbose=True

def prompter(test_instance, step = 1, gestalt=False):
    """
    Generate formatted strings based on the provided dictionary containing true and false examples,
    with true and false examples alternating. Also includes a test case and its label.

    Parameters:
    test_instance (dict): Dictionary containing keys 'true_examples', 'false_examples', 'test', and 'label'.

    Returns:
    str: Formatted string for each example and test case.
    """
    output = []
    
    # Iterate over the maximum length of true or false examples
    for i in range(max(len(test_instance['true_examples']), len(test_instance['false_examples']))):
        if i < len(test_instance['true_examples']):
            output.append(f'Input: "{test_instance["true_examples"][i]}" Label: True')
        if i < len(test_instance['false_examples']):
            output.append(f'Input: "{test_instance["false_examples"][i]}" Label: False')
    
    # Add the test case with its label
    test_label = 'True' if test_instance['label'] else 'False'
    if step == 1:
        output.append(f'\nInput: "{test_instance["test"]}" Label: \nTrue \nFalse')
    elif step == 2 and gestalt==False:
        output.append(f'\n\nWrite rule(s) to label True or False.\n\n(Response Structure)\nRule:\n...')
    elif step == 2 and gestalt==True:
        output.append(f'\n\nFollow these steps:\n1. Group True and False examples.\n2. Write down an exhaustive list of common characteristics in True examples.\n3. Write down an exhaustive list of common characteristics in False examples.\n4. Identify the diverging pattern between the two groups.\n5. Revise and choose the most striking difference between True and False.\n6. Write rule(s) to label True or False, with the most striking difference above.\n\n(Response Structure)\nStep 1: Group True and False Examples\n...\nStep 2: Common Characteristics in True Examples\n...\nStep 3: Common Characteristics in False Examples\n...\nStep 4: Differences between True and False Examples\n...\nStep 5: The Most Consistent Difference between True and False\n...\nStep 6: A Very Detailed Manual for Labeling True or False \n...')

    return '\n'.join(output)

def gestalt_prompter(test_instance):
    """
    Generate formatted strings based on the provided dictionary containing true and false examples,
    with true and false examples alternating. Also includes a test case and its label.

    Parameters:
    test_instance (dict): Dictionary containing keys 'true_examples', 'false_examples', 'test', and 'label'.

    Returns:
    str: Formatted string for each example and test case.
    """
    output = []
    
    # Iterate over the maximum length of true or false examples
    for i in range(max(len(test_instance['true_examples']), len(test_instance['false_examples']))):
        if i < len(test_instance['true_examples']):
            output.append(f'Input: "{test_instance["true_examples"][i]}" Label: True')
        if i < len(test_instance['false_examples']):
            output.append(f'Input: "{test_instance["false_examples"][i]}" Label: False')
    
    # Add the test case with its label
    test_label = 'True' if test_instance['label'] else 'False'
    output.append(f'\n\nFollow these steps:\n1. Group True and False examples.\n2. Find common characteristics in True examples.\n3. Find common characteristics in False examples.\n4. Find the diverging pattern between the two groups.\n5. Write a set of steps to follow to label True or False\n6. Using the rules, run all given examples and record accuracy\n7. If the accuracy is below 90%, start again from step 2\n...(Retry...)...\nNow, answer this -> Input: "{test_instance["test"]}" Label: \nTrue \nFalse')

    return '\n'.join(output)

def get_response(model: str, 
                 messages: List[dict],
                 max_tokens: Optional[int] = None, 
                 temperature: Optional[float] = None, 
                 top_p: Optional[float] = None, 
                 n: Optional[int] = None, 
                 presence_penalty: Optional[float] = None, 
                 frequency_penalty: Optional[float] = None, 
                 stop: Optional[List[str]] = None,
                 other_params: Optional[Dict] = None
                ) -> str:
    """
    Get a response from the LiteLLM API.

    Args:
    model (str): Model name, e.g., "gpt-3.5-turbo".
    messages (list): messages for the model.
    max_tokens (int, optional): Maximum number of tokens to generate.
    temperature (float, optional): Sampling temperature.
    top_p (float, optional): Nucleus sampling parameter.
    n (int, optional): Number of completions to generate.
    presence_penalty (float, optional): Presence penalty parameter.
    frequency_penalty (float, optional): Frequency penalty parameter.
    stop (list of str, optional): Sequences where the API will stop generating further tokens.
    other_params (dict, optional): Any other provider-specific parameters.

    Returns:
    str: The generated text response from the model.
    """

    # Combine all parameters for the API call
    api_params = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "n": n,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "stop": stop,
        "num_retries" : 50
    }
    if other_params:
        api_params.update(other_params)

    response = completion(**api_params)#, image = 'https://upload.wikimedia.org/wikipedia/commons/b/b1/Black_Screen.jpg')

    return response.choices[0].message.content if response.choices else "No response generated"

def determine_binary_answers(model_response):
    """
    Determines whether each string in a list indicates answer 'A' or 'B', accounting for various formats.

    Args:
    model_response: model_response containing the answers.

    Returns:
    a, b, or n
    """
    # Normalize string to handle case and whitespace variations
    normalized_string = model_response.strip().lower()

    # Check various cases
    if "true" in normalized_string and "false" not in normalized_string or 'label: true' in normalized_string:
        return True
    elif "false" in normalized_string and "true" not in normalized_string or 'label: false' in normalized_string:
        return False
    else:
        print(f"Non-binary::: {normalized_string}")
        return None