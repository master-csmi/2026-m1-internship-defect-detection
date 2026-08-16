# AI assistance notes

This folder records where Artificial Intelligence tools were used in this project.
It exists because the CSMI policy asks for traceability of AI use.

The rule I followed: a tool may draft, it may not decide. Everything that came out
of a tool was read, tested and adapted before it entered the repository, and no
number in the report comes from a model.

## Tools used

| Tool | Used for |
|---|---|
| Claude (Anthropic) | Reformulating sentences in the report; help with the deep learning part; correcting code written earlier, and help me draft the tests |
| ChatGPT (OpenAI) | Explanations of algorithms; help with LaTeX formatting |
| Google Gemini | Explanations of medical and physiological ECG vocabulary |



## How the assistance was checked

Code that was drafted or corrected with assistance is covered by the test suite in
`tests/`, which checks properties rather than remembered outputs: shapes,
mathematical identities such as the wavelet energy ratios summing to one, and
behaviour on toy signals where the right answer is known in advance. The tests run
automatically on every change through GitHub Actions.

Report text was checked against the results files in `results/`. Every table in the
report is copied from a `comparison_table.csv` written by a notebook, every figure
from the `figures/` folders, and every reference cited was opened and read.

## What was not done with AI assistance

- The choice of dataset, of the evaluation protocol and of the DS1/DS2 split.
- The design of the 35 features and the choice of which families to include.
- Every experimental result. All numbers come from running the code.
- The interpretation of the results and the conclusions drawn from them.
