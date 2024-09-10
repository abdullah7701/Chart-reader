# Chart Q&A Challenge

## Overview

This challenge focuses on solving various types of questions related to chart data. The task is to build a solution that can accurately answer questions regarding the highest and lowest values, differences between values, and comparisons over time. Your goal is to achieve an average score (`avg_score`) of at least 0.9 across 1,000 test questions by implementing or replacing your model/solution in the `main.py` file.

## Question Types

Currently, there are 5 types of questions to be addressed:

1. **What is the highest value of {label}?**  
   - Determine the highest recorded value of the given label.

2. **What is the lowest value of {label}?**  
   - Identify the lowest recorded value of the given label.

3. **What is the difference between the high and low of {label}?**  
   - Calculate the difference between the maximum and minimum values of the given label.

4. **On what date are {label1} and {label2} closest in value?**  
   - Find the date on which the two given labels have the smallest difference in their values.

5. **On what day is the largest difference between {label1} and {label2}?**  
   - Identify the day when the difference between the two given labels is the greatest.

## Challenge Objective

- Solve 1,000 questions of the types mentioned above.
- Achieve an average score (`avg_score`) of **0.9** or higher.
- Replace or implement your model/solution in the `main.py` file to handle the question types effectively.

## Instructions

1. **Setup**: Clone the repository and ensure that all dependencies are installed.
2. **Replace Your Model/Solution**: Implement your model or solution in `main.py`. The file contains a placeholder for your code. Make sure your solution can handle all 5 types of questions effectively.
3. **Testing and Evaluation**: Run the test suite to validate your solution against 1,000 questions. The goal is to achieve an `avg_score` of at least 0.9.
4. **Submit**: Once you achieve the target score, submit your solution for review.

## Scoring

- Your solution will be evaluated based on its ability to correctly answer the questions with high accuracy.
- The target is to maintain an `avg_score` of **0.9** or above.

## Getting Started

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
2. Replace the placeholder code in main.py with your solution.
3. Run the evaluation script:
   ```bash
   python main.py
   ```
## Tips for Success

- Ensure your model is well-tuned for the types of questions being asked.
- Consider edge cases such as ties, missing data, and date formatting issues.
- Optimize for performance, as the solution should handle a large number of queries efficiently.
