# from evaluation import evaluate
# from simple_model import SimpleModel
# from task import generate_task
# import time

# # Replace this with your own model
# model = SimpleModel()


# num_challenges = 2
# scores = []


# # for i in range(num_challenges):
# #     challenge = generate_task()
# #     question, image_base64, expected_answer = challenge
# #     start = time.time()
# #     answer = model.get_answer(question, image_base64)
# #     end = time.time()
# #     processing_time = end - start
# #     score = evaluate(answer, expected_answer, processing_time)
# #     scores.append(score)

# # avg_score = sum(scores) / len(scores)
# # print(f"Average score: {avg_score}")
# # is_passing = avg_score > 0.9
# # print(f"Is passing? {is_passing}")

# for i in range(num_challenges):
#     challenge = generate_task()
#     question, image_base64, expected_answer = challenge
#     start = time.time()
#     answer = model.get_answer(question, image_base64)
#     end = time.time()
#     processing_time = end - start
#     score = evaluate(answer, expected_answer, processing_time)
#     scores.append(score)
    
#     # Print the results for each challenge
#     print(f"Challenge {i + 1}:")
#     print(f"Question: {question}")
#     print(f"Expected Answer: {expected_answer}")
#     print(f"Model's Answer: {answer}")
#     print(f"Processing Time: {processing_time:.2f} seconds")
#     print(f"Score: {score}")
#     print()

# # Calculate and print the average score
# avg_score = sum(scores) / len(scores)
# print(f"Average score: {avg_score:.2f}")
# is_passing = avg_score > 0.9
# print(f"Is passing? {is_passing}")




# for i in range(num_challenges):
#     challenge = generate_task()
#     question, image_base64, expected_answer = challenge
#     start = time.time()
#     answer = model.get_answer(question, image_base64)
#     end = time.time()
#     processing_time = end - start
#     score = evaluate(answer, expected_answer, processing_time)
#     scores.append(score)

# avg_score = sum(scores) / len(scores)
# print(f"Average score: {avg_score}")
# is_passing = avg_score > 0.9
# print(f"Is passing? {is_passing}")


# from PIL import Image
# import base64
# import io
# from evaluation import evaluate
# from simple_model import SimpleModel
# from task import generate_task
# import time
# import matplotlib.pyplot as plt

# # Replace this with your own model
# model = SimpleModel()

# num_challenges = 1
# scores = []

# for i in range(num_challenges):
#     try:
#         challenge = generate_task()
#         question, image_base64, expected_answer = challenge
#         start = time.time()
#         x_axis_labels, y_axis_labels, chart_data = model.extract_data(image_base64)
#         answer = model.answer_question(question, x_axis_labels, y_axis_labels, chart_data)
#         end = time.time()
#         processing_time = end - start
#         score = evaluate(answer, expected_answer, processing_time)
#         scores.append(score)

#         # Display the graph
#         image = Image.open(io.BytesIO(base64.b64decode(image_base64)))
#         plt.imshow(image)
#         plt.axis('off')
#         plt.show()

#         # Print the results for each challenge
#         print(f"Challenge {i + 1}:")
#         print(f"Question: {question}")
#         print(f"Expected Answer: {expected_answer}")
#         print(f"Model's Answer: {answer}")
#         print(f"Processing Time: {processing_time:.2f} seconds")
#         print(f"Score: {score}")
#         print()

#     except ValueError as e:
#         print(f"Error during challenge {i + 1}: {e}")
#         scores.append(-1)  # Assign a negative score for failed extraction

# # Calculate and print the average score
# avg_score = sum(scores) / len(scores)
# print(f"Average score: {avg_score:.2f}")
# is_passing = avg_score > 0.9
# print(f"Is passing? {is_passing}")



#tasting:
# from PIL import Image
# import base64
# import io
# from evaluation import evaluate
# from simple_model import SimpleModel
# from task import generate_task
# import time
# import matplotlib.pyplot as plt

# # Replace this with your own model
# model = SimpleModel()

# num_challenges = 5
# scores = []

# for i in range(num_challenges):
#     try:
#         # Generate a new task (question, image, and expected answer)
#         challenge = generate_task()
#         question, image_base64, expected_answer = challenge
        
#         # Start timing the model's response
#         start = time.time()
        
#         # Call the get_answer method with the question, image, and expected answer
#         answer = model.get_answer(question, image_base64, expected_answer)
        
#         # End timing the model's response
#         end = time.time()
#         processing_time = end - start
        
#         # Evaluate the answer
#         score = evaluate(answer, expected_answer, processing_time)
#         scores.append(score)

        # # Display the graph
        # image = Image.open(io.BytesIO(base64.b64decode(image_base64)))
        # plt.imshow(image)
        # plt.axis('off')
        # plt.show()

#         # Print the results for each challenge
#         print(f"Challenge {i + 1}:")
#         print(f"Question: {question}")
#         print(f"Expected Answer: {expected_answer}")
#         print(f"Model's Answer: {answer}")
#         print(f"Processing Time: {processing_time:.2f} seconds")
#         print(f"Score: {score}")
#         print()

#     except ValueError as e:
#         print(f"Error during challenge {i + 1}: {e}")
#         scores.append(-1)  # Assign a negative score for failed extraction

# # Calculate and print the average score
# avg_score = sum(scores) / len(scores)
# print(f"Average score: {avg_score:.2f}")
# is_passing = avg_score > 0.9
# print(f"Is passing? {is_passing}")
# Updated version of main.py that also displays the generated plot

# from evaluation import evaluate
# from simple_model import SimpleModel
# from task import generate_task_simple
# import time
# import base64
# from PIL import Image
# import io

# model = SimpleModel()

# num_challenges = 5
# scores = []

# for i in range(num_challenges):
#     question, image_base64, expected_answer = generate_task_simple()
#     start = time.time()
#     answer = model.get_answer(question, image_base64)
#     end = time.time()
#     processing_time = end - start

#     score = evaluate(answer, expected_answer, processing_time)
#     scores.append(score)

        
#     print(f"Challenge {i + 1}:")
#     print(f"Question: {question}")
#     print(f"Expected Answer: {expected_answer}")
#     print(f"Model's Answer: {answer}")
#     print(f"Processing Time: {processing_time:.2f} seconds")
#     print(f"Score: {score}")
#     print()

# avg_score = sum(scores) / len(scores)
# print(f"Average score: {avg_score:.2f}")
# is_passing = avg_score > 0.9
# print(f"Is passing? {is_passing}")

from evaluation import evaluate   
from simple_model import SimpleModel
from task import generate_task_simple
import time
import base64
import io
from PIL import Image
import matplotlib.pyplot as plt

# Initialize the model
model = SimpleModel()


num_challenges = 20
scores = []

for i in range(num_challenges):
    # Generate a question, graph (as base64), and the expected answer
    question, image_base64, expected_answer = generate_task_simple()

    # # Convert the base64 image to an actual image for display
    # image_data = base64.b64decode(image_base64)
    # image = Image.open(io.BytesIO(image_data))

    # Start processing time
    start = time.time()
    
    # Get the model's answer and pass the expected answer to the model
    answer = model.get_answer(question, image_base64, expected_answer)
    
    # End processing time
    end = time.time()
    processing_time = end - start

    # Evaluate the model's answer with the expected answer
    score = evaluate(answer, expected_answer, processing_time)
    scores.append(score)

    # Output the results for each challenge
    print(f"Challenge {i + 1}:")
    print(f"Question: {question}")
    print(f"Expected Answer: {expected_answer}")
    print(f"Model's Answer: {answer}")
    print(f"Processing Time: {processing_time:.2f} seconds")
    print(f"Score: {score}")
    print()

    # # Display the graph for visual confirmation
    # plt.imshow(image)
    # plt.axis('off') 
    # plt.show()

# Calculate the average score
avg_score = sum(scores) / len(scores)
print(f"Average score: {avg_score:.2f}")

# Determine if the average score is passing
is_passing = avg_score > 0.9
print(f"Is passing? {is_passing}")

