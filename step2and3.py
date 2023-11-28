import os
import pandas as pd
from utils import prompter, get_response, determine_binary_answers, gestalt_prompter
from tqdm import tqdm
from pprint import pprint
from collections import defaultdict
os.environ["OPENAI_API_KEY"] = "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"
os.environ["COHERE_API_KEY"] = "j47X9Yy01ChYJiHaFcpUwmKE0MEYVeXgCu2bTu7m"
os.environ["HUGGINGFACE_API_KEY"] = "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" 
os.environ['AI21_API_KEY'] = "34y3ctmr1zyg2ofCZe85sI74r6Z0Tlf9"
os.environ["REPLICATE_API_KEY"] = "r8_cocqTHulK4QBNcg8S6vc3NBvNuoCEab3VFPId"

"""Stage 1
Set Hyperparameters and Load Data
"""
k_true = 10 # how many true examples to show?
k_false = 10 # how many false examples to show?
rule_articulator = "gpt-4-0613"
rule_verifiers = ["gpt-3.5-turbo-0613"]
model_temperature = 0.0001
gestalt_prompting = False

# Load the dataset
file_path = 'dataset.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
print(data.head())



"""Stage 2
Process Test Data

Our goal is to create a list of dictionary where each dictionary is a test instance and looks like below at k-shot=20 setup
{
    'true_examples': [
        'DREAM BIG AND DARE TO FAIL.', 'SMILE, BREATHE, AND GO SLOWLY.', 'WHEREVER YOU GO, GO WITH ALL YOUR HEART.',
        'DO WHAT YOU CAN WITH ALL YOU HAVE, WHEREVER YOU ARE.', "BELIEVE YOU CAN AND YOU'RE HALFWAY THERE.",
        'THE ONLY WAY TO DO GREAT WORK IS TO LOVE WHAT YOU DO.', 'IF YOU WANT TO LIVE A HAPPY LIFE, TIE IT TO A GOAL, NOT TO PEOPLE OR THINGS.',
        'THE FUTURE BELONGS TO THOSE WHO BELIEVE IN THE BEAUTY OF THEIR DREAMS.', 'THE PURPOSE OF OUR LIVES IS TO BE HAPPY.',
        "LIFE IS WHAT HAPPENS WHEN YOU'RE BUSY MAKING OTHER PLANS."
    ],
    'false_examples': [
        'Get busy living or get busy dying.', "Whether you think you can or you think you can't, you're right.",
        'The only impossible journey is the one you never begin.', 'In order to write about life first, you must live it.',
        'The big lesson in life, baby, is never be scared of anyone or anything.', "Sing like no one's listening, love like you've never been hurt, dance like nobody's watching.",
        'Curiosity about life in all of its aspects, I think, is still the secret of great creative people.', 'Life is not a problem to be solved, but a reality to be experienced.',
        'The healthiest response to life is joy.', 'Life is really simple, but men insist on making it complicated.'
    ],
    'test': 'A QUICK BROWN FOX JUMPS OVER THE LAZY DOG.',
    'label': True,
    'seed_rule' : 'The input is all uppercase.'
}
"""
# There are 20 rules tested, 10 test instances for each rule, making 200 test instances in total
test_instances = []
for rule_idx in range(1,21):
    rule_data = data.iloc[rule_idx - 1]

    # Prepare true and false examples
    true_examples = [rule_data[f"True_Example_{i}"] for i in range(1, k_true+1)]
    false_examples = [rule_data[f"False_Example_{i}"] for i in range(1, k_false+1)]

    true_tests = [rule_data[f"True_Test_{i}"] for i in range(1, 6)]
    false_tests = [rule_data[f"False_Test_{i}"] for i in range(1, 6)]

    # Each run of this loop should add five test instances
    for true_test in true_tests:
        test_instances.append({
            "true_examples" : true_examples,
            "false_examples" : false_examples,
            "test" : true_test,
            "label" : True,
            "seed_rule" : rule_data["Seed_Rule"],
            "rule_idx" : rule_idx
        })
    
    # Each run of this loop should add five test instances
    for false_test in false_tests:
        test_instances.append({
            "true_examples" : true_examples,
            "false_examples" : false_examples,
            "test" : false_test,
            "label" : False,
            "seed_rule" : rule_data["Seed_Rule"],
            "rule_idx" : rule_idx
        })
    

"""Stage 3
Run Test
"""
# For every unique seed rule (which means unique set of examples), articulate rules.
#target_seed_rules = ['The input is all uppercase.', 'The input ends with a punctuation mark.', 'The input is a palindrome (reads the same forwards and backwards).', 'The input contains more vowels than consonants.', 'The input includes a proper noun.', 'The input includes a number spelled out (e.g., "five").', 'The input contains a rhyming pair of words.', 'The input includes a city or country name.', 'The input contains a word repeated consecutively (e.g., "very very").', 'The input includes an animal name.', 'The input contains a word with a hyphen (e.g., "well-known").', 'The input includes a form of address (e.g., Mr., Dr.).']
target_seed_rules = ["The input is all uppercase.","The input starts with a vowel.","The input ends with a punctuation mark.","The input contains exactly three words.","The input is a palindrome (reads the same forwards and backwards).","The input contains more vowels than consonants.","The input includes a color name.","The input has an even number of alphanumeric characters.","The input includes a proper noun.","The input contains a verb in past tense.","The input contains a word longer than 7 letters.","The input includes a number spelled out (e.g., \"five\").","The input contains a rhyming pair of words.","The input includes a city or country name.","The input contains a word repeated consecutively (e.g., \"very very\").","The input includes an animal name.","The input contains a mathematical symbol (e.g., %, =, +).","The input contains a word with a hyphen (e.g., \"well-known\").","The input includes a form of address (e.g., Mr., Dr.).","The input contains a word with silent letters (e.g., \"knight\")."]
target_seed_rules_with_articulated_rules = {}
for test_instance in tqdm(test_instances):
    if test_instance['seed_rule'] in target_seed_rules:
        if gestalt_prompting == False:
            articulated_rule = get_response(
                model = rule_articulator,
                messages = [{"content": prompter(test_instance, step=2), "role": "user"}],
                temperature = model_temperature 
            )
            target_seed_rules_with_articulated_rules[test_instance['seed_rule']] = {
                    'articulated_rule': articulated_rule,
                    "true_examples" : test_instance['true_examples'],
                    "false_examples" : test_instance['false_examples'],
                }
            target_seed_rules.remove(test_instance['seed_rule'])
        else:
            articulated_rule = get_response(
                model = rule_articulator,
                messages = [{"content": prompter(test_instance, step=2, gestalt=True), "role": "user"}],
                temperature = model_temperature 
            )
            articulated_rule = articulated_rule.split('A Very Detailed Manual for Labeling True or False')[1]
            print(articulated_rule)
            target_seed_rules_with_articulated_rules[test_instance['seed_rule']] = {
                    'articulated_rule': articulated_rule,
                    "true_examples" : test_instance['true_examples'],
                    "false_examples" : test_instance['false_examples'],
                }
            target_seed_rules.remove(test_instance['seed_rule'])

# Now that we have collected articulated rules in target_seed_rules_with_articulated_rules, we proceed to test it with rule_verifiers
def nested_defaultdict():
    return defaultdict(int)

# Testing the articulated rules on test instances
result_test = defaultdict(lambda: defaultdict(nested_defaultdict))
for rule_verifier in rule_verifiers:
    for test_instance in tqdm(test_instances):
        try:
            articulated_rule = target_seed_rules_with_articulated_rules[test_instance["seed_rule"]]['articulated_rule']
        except KeyError: 
            continue
            
        prediction_response = get_response(
            model = rule_articulator,
            messages = [{
                "content": f'{articulated_rule} \nInput: "{test_instance["test"]}" Label: \nTrue \nFalse', 
                "role": "user"}],
            temperature = model_temperature 
        )
        pred = determine_binary_answers(prediction_response)

        if test_instance['label'] == pred:
            result_test[rule_verifier][test_instance["seed_rule"]]['correct'] += 1
        else:
            result_test[rule_verifier][test_instance["seed_rule"]]['wrong'] += 1
pprint(result_test)

# Testing the articulated rules on example instances
result_example = defaultdict(lambda: defaultdict(nested_defaultdict))
for rule_verifier in rule_verifiers:
    for seed_rule in target_seed_rules_with_articulated_rules:
        articulated_rule = target_seed_rules_with_articulated_rules[seed_rule]['articulated_rule']
        true_examples = target_seed_rules_with_articulated_rules[seed_rule]['true_examples']
        false_examples = target_seed_rules_with_articulated_rules[seed_rule]['false_examples']

        for true_example in true_examples:
            prediction_response = get_response(
                model = rule_articulator,
                messages = [{
                    "content": f'{articulated_rule} \nInput: "{true_example}" Label: \nTrue \nFalse', 
                    "role": "user"}],
                temperature = model_temperature 
            )
            pred = determine_binary_answers(prediction_response)
            if pred == True:
                result_example[rule_verifier][seed_rule]['correct'] += 1
            else:
                result_example[rule_verifier][seed_rule]['wrong'] += 1

        for false_example in false_examples:
            prediction_response = get_response(
                model = rule_articulator,
                messages = [{
                    "content": f'{articulated_rule} \nInput: "{false_example}" Label: \nTrue \nFalse', 
                    "role": "user"}],
                temperature = model_temperature 
            )
            pred = determine_binary_answers(prediction_response)
            if pred == False:
                result_example[rule_verifier][seed_rule]['correct'] += 1
            else:
                result_example[rule_verifier][seed_rule]['wrong'] += 1

pprint(result_example)