README FOR THE PAPER "Attentive Reasoning Queries: A Systematic Method for Optimizing Instruction-Following in Large Language Models"
FOR INSTALLATION INSTURCTIONS AND OTHER INFORMATION ABOUT PARLANT, VISIT OUR MAIN BRANCH

SUPPLEMENTARY MATERIALS:
All supplementary materials, including:
- example prompts for ARQ, CoT and no reasoning (control) are available under supplementary_materials/prompts/
- plots and figures are available under supplementary_materials/plots/


EXPERIMENT:
The main experiment was performed by running all tests in this branch's test suite. Whenever a test is ran, its results are saved in a local jsonl file named "parlant_test_results.jsonl".
The python code in "analyze_results" is then used to produce the paper's tables from this jsonl file.

To run all tests, install Parlant, switch to this branch, and run the command "pytest --no-cache".
To analyze multiple test runs, it is recommended to install "pytest-repeat" and run "pytest --no-cache --count=<desired number>".

Note that tests testing the guideline proposer exclusively are under test_guideline_proposer.
All other tests are considered comprehensive, and are implemented using Behavior Driven Development (BDD) in dedicated gherkin files.









Questions and comments are welcome at bar@emcie.co and on our discord server
