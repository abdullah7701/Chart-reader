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



FOR MAX ACCURACY:
It will take Less time when Run on Colab because of Laptop Processing power:

Challenge 1:
Question: What is the lowest value?
Expected Answer: 13.39
Model's Answer: 13.39
Processing Time: 5.24 seconds
Score: 0.9545454545454546

Challenge 2:
Question: What is the lowest value?
Expected Answer: 11.95
Model's Answer: 11.95
Processing Time: 4.68 seconds
Score: 1.0

Challenge 3:
Question: What is the lowest value?
Expected Answer: 10.33
Model's Answer: 10.33
Processing Time: 4.56 seconds
Score: 1.0

Challenge 4:
Question: What is the lowest value?
Expected Answer: 10.96
Model's Answer: 10.96
Processing Time: 5.63 seconds
Score: 0.9545454545454546

Challenge 5:
Question: What is the difference between the high and low values?
Expected Answer: 80.32
Model's Answer: 80.32
Processing Time: 9.31 seconds
Score: 0.8636363636363636

Challenge 6:
Question: What is the lowest value?
Expected Answer: 12.19
Model's Answer: 12.19
Processing Time: 7.68 seconds
Score: 0.9272727272727272

Challenge 7:
Question: What is the lowest value?
Expected Answer: 15.05
Model's Answer: 15.05
Processing Time: 5.08 seconds
Score: 1.0

Challenge 8:
Question: What is the lowest value?
Expected Answer: 13.48
Model's Answer: 13.48
Processing Time: 5.05 seconds
Score: 1.0

Challenge 9:
Question: What is the highest value?
Expected Answer: 96.36
Model's Answer: 96.36
Processing Time: 5.71 seconds
Score: 0.9545454545454546

Challenge 10:
Question: What is the difference between the high and low values?
Expected Answer: 86.78
Model's Answer: 86.78
Processing Time: 11.10 seconds
Score: 0.8636363636363636

Challenge 11:
Question: What is the highest value?
Expected Answer: 81.45
Model's Answer: 81.88
Processing Time: 1.14 seconds
Score: 0.9090909090909091

Challenge 12:
Question: What is the highest value?
Expected Answer: 97.07
Model's Answer: 97.07
Processing Time: 6.56 seconds
Score: 0.9545454545454546

Challenge 13:
Question: What is the difference between the high and low values?
Expected Answer: 74.99
Model's Answer: 74.99
Processing Time: 9.49 seconds
Score: 0.8636363636363636

Challenge 14:
Question: What is the difference between the high and low values?
Expected Answer: 84.62
Model's Answer: 84.62
Processing Time: 8.61 seconds
Score: 0.9272727272727272

Challenge 15:
Question: What is the highest value?
Expected Answer: 89.45
Model's Answer: 81.90
Processing Time: 1.53 seconds
Score: 0.36363636363636365

Challenge 16:
Question: What is the difference between the high and low values?
Expected Answer: 87.40
Model's Answer: 87.40
Processing Time: 6.57 seconds
Score: 0.9545454545454546

Challenge 17:
Question: What is the difference between the high and low values?
Expected Answer: 77.29
Model's Answer: 77.29
Processing Time: 10.16 seconds
Score: 0.8636363636363636

Challenge 18:
Question: What is the highest value?
Expected Answer: 98.15
Model's Answer: 98.15
Processing Time: 5.33 seconds
Score: 0.9545454545454546

Challenge 19:
Question: What is the highest value?
Expected Answer: 99.44
Model's Answer: 99.44
Processing Time: 8.47 seconds
Score: 0.9272727272727272

Challenge 20:
Question: What is the highest value?
Expected Answer: 93.82
Model's Answer: 93.82
Processing Time: 6.06 seconds
Score: 0.9545454545454546

Average score: 0.91
Is passing? True


COLAB's Reuslt:
![image](https://github.com/user-attachments/assets/a17abbc5-1a23-4d0f-b3a8-31b42ff34e33)



______________________________________________________________________________________
Requirements
Ensure you have all necessary dependencies installed
pip install -r requirements.txt


How to Use
Run the Model: To use the SimpleModel, call the get_answer() function by passing the question, chart image (in base64), and expected answer (optional).

Sanmple code to Run the Simple_model.py:
""
from simple_model import SimpleModel

model = SimpleModel()

question = "What is the highest value?"
image_base64 = "<your_base64_image_string>"
answer = model.get_answer(question, image_base64)
print("Model's Answer:", answer)

""
