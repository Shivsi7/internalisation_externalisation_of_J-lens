Homework zero -- as part of application to take the Harvard CS 2881 AI Safety course

Relationship between J-space and model chain of thought
Due: August 5, 2026, at 11:59 p.m. Eastern Time

Expected time commitment: approximately 4–16 hours

Deliverables: a report of approximately two pages and code in a GitHub repository

Submitting Homework Zero is a necessary but not sufficient condition for admission. Completing the assignment does not guarantee admission to the course.



Submission and AI-use information
Submit a report of approximately two pages together with a link to a GitHub repository containing your code using the Homework Zero submission form.
You are encouraged to use AI coding agents, generative AI tools, or anything else that helps you complete the assignment. You remain responsible for understanding, checking, and clearly explaining the work you submit.
Harvard students working on Homework Zero will be reimbursed for up to $100 of AI costs used to complete it. Reimbursement details will be provided separately.
Questions about the course or Homework Zero should be sent to cs2881@boazbarak.org.
Repository requirements
Your GitHub repository should be self-contained and organized so that we can understand your work, locate every component, and reproduce the main results from the repository alone.

Make the repository private and add the lecturer and TFs as collaborators before submitting it.
Place your report in the root directory as report.pdf.
Include a README.md in the root directory. It should explain the repository’s structure and where everything is located, provide setup and reproduction instructions, and include links to all relevant external URLs.
Include the code, prompts, configuration files, experimental settings, commands or scripts, and results needed to understand and reproduce your experiment. The README should explain how to regenerate the principal tables and figures in the report.
If a necessary artifact cannot reasonably be stored in the repository, link to it from the README and explain exactly how to obtain and use it.
Read
Verbalizable Representations Form a Global Workspace in Language Models (Anthropic, Transformer Circuits, July 2026).

Pay particular attention to the section “The J-space selectively mediates flexible but not automatic cognition,” especially the subsection “J-space ablation leaves most capabilities intact while impairing internal reasoning.”

Background
The paper introduces the Jacobian lens (J-lens), a technique that surfaces the set of “verbalizable” representations a model is poised to put into words. Collectively, these representations form the J-space, which the authors argue functions like a cognitive global workspace: a small, privileged set of representations that the model can report, reason with, and manipulate, sitting atop a much larger volume of automatic processing.

One result matters most for this assignment. When the authors ablate the J-space—at each token position, across the workspace layer band, they zero the residual stream’s projection onto the top-k active J-lens vectors—most capabilities survive, including parsing, classification, and one-step recall, but internal multi-step reasoning collapses.

Critically, they find that GSM8K solved with explicit chain of thought is far more robust to this ablation than the same problems answered directly. They interpret this as the model externalizing onto the page what it would otherwise have to hold in its internal workspace: written chain of thought and the internal J-space appear to be partially interchangeable “scratchpads.”

Your task
Investigate how this internal–external trade-off behaves, and whether it continues to hold as problems get harder.

Constraints
Model: Qwen3-4B
Datasets: GSM8K, MATH-500, and AIME. These span a wide difficulty range on purpose. Clearly state the exact AIME set and source you use.
Expected time commitment: approximately 4–16 hours
Report: approximately two pages, plus code in a GitHub repository
What we are looking for
We are not looking for a particular “correct” conclusion. We want to see how you approach designing experiments to investigate the question.

Before you run anything, write down what you expect to happen and why. Then design the experiment so that the opposite result would also have been visible if it were true. A clean answer to a question designed only to confirm your initial expectation is worth less than a messy, honest answer to a question you set up fairly.

Your comparison should make it possible to distinguish the effects of J-space ablation from broad model degradation. Explain your controls, evaluation choices, and any implementation or compute limitations. Report negative and inconclusive results as clearly as positive ones.

Report structure
Your report should have the following sections:

Hypothesis. State a hypothesis about how ablating the J-space may trade off against using external chain of thought. Record this before running the main experiments.
Experiment design. Describe the experiment or experiments you designed to test the hypothesis and explain why they are relevant. Include the conditions, comparisons, controls, and metrics needed for either confirming or disconfirming evidence to be visible.
Experimental details. Explain what you did and how you ran the experiment. Include enough information to understand the implementation and reproduce the main results.
Experimental results. Present the results, including plots, tables, and other relevant information. Include uncertainty, failures, or missing runs where appropriate.
Analysis of results. Explain what the results say about your hypothesis, what alternative explanations remain, and what you would test next with more time or compute.
