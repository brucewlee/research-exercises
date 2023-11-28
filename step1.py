import os
import pandas as pd
from utils import prompter, get_response, determine_binary_answers, gestalt_prompter
from tqdm import tqdm
from pprint import pprint
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
model_to_test = "gpt-3.5-turbo-0301"
#model_to_test = 'huggingface/dandelin/vilt-b32-finetuned-vqa'
#model_to_test = "replicate/deployments/brucewlee/demo-3"
model_temperature = 1

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
    
# 200 test instances
print(f"number of test instances :\n{len(test_instances)}\n")
print(f"example of a test instance :\n{test_instances[0]}\n")


"""Stage 3
Run Test
"""
result = []
for test_instance in tqdm(test_instances):
    prediction_response = get_response(
            model = model_to_test, 
            messages = [{"content": prompter(test_instance), "role": "user"}], 
            temperature = model_temperature
        )
    pred = determine_binary_answers(prediction_response)

    if test_instance['label'] == pred:
        result.append((test_instance["seed_rule"], 1))
    else:
        result.append((test_instance["seed_rule"], 0))



"""Stage 4
Analyze Result
"""
# Initialize a dictionary to store counts
category_counts = {}

# Iterate over the data
for category, correct in result:
    if category not in category_counts:
        category_counts[category] = {'total': 0, 'correct': 0}
    category_counts[category]['total'] += 1
    if correct == 1:
        category_counts[category]['correct'] += 1

# Calculate average accuracy per category
accuracy_per_category = {}
for category, counts in category_counts.items():
    accuracy = counts['correct'] / counts['total']
    accuracy_per_category[category] = accuracy

pprint(accuracy_per_category)