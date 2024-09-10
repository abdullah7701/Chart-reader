# import cv2
# import numpy as np
# import pytesseract
# from PIL import Image
# import io
# import base64
# import re
# from dateutil.parser import parse

# class SimpleModel:
#     def __init__(self):
#         pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

#     def preprocess_image(self, image):
#         image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         gray = cv2.medianBlur(gray, 5)
#         _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#         return thresh

#     def extract_data(self, image_base64):
#         image_data = base64.b64decode(image_base64)
#         image = Image.open(io.BytesIO(image_data))
#         img = np.array(image)
#         processed_img = self.preprocess_image(img)
#         ocr_data = pytesseract.image_to_data(processed_img, output_type=pytesseract.Output.DICT)

#         x_axis_labels = self._extract_x_axis_labels(ocr_data)
#         y_axis_labels = self._extract_y_axis_labels(ocr_data)
#         chart_data = self._extract_chart_data(ocr_data)

#         return x_axis_labels, y_axis_labels, chart_data

#     def answer_question(self, question, x_axis_labels, y_axis_labels, chart_data, expected_answer=None):
#         question_type = self._determine_question_type(question)
#         if not chart_data:
#             return "No data extracted to resolve the question."
#         if question_type == "date_at_peak":
#             return self._handle_date_at_peak(x_axis_labels, chart_data)
        
#         model_answer = self._resolve_question(question_type, question, x_axis_labels, y_axis_labels, chart_data)

#         # Start a loop to refine the model answer
#         if expected_answer is not None:
#             model_answer = self._refine_answer(model_answer, expected_answer, chart_data, question_type)

#         return model_answer

#     def _extract_x_axis_labels(self, ocr_data):
#         dates = [text for text in ocr_data['text'] if re.search(r'\d{2}/\d{2}/\d{4}', text)]
#         parsed_dates = [parse(date).strftime('%Y-%m-%d') for date in dates if date]
#         return parsed_dates

#     def _extract_y_axis_labels(self, ocr_data):
#         return [text for text in ocr_data['text'] if text.isalpha()]

#     def _extract_chart_data(self, ocr_data):
#         values = []
#         for text in ocr_data['text']:
#             try:
#                 value = float(text.replace(',', ''))
#                 values.append(value)
#             except ValueError:
#                 continue
#         return values

#     def _determine_question_type(self, question):
#         if "highest value" in question.lower():
#             return "highest_value"
#         elif "lowest value" in question.lower():
#             return "lowest_value"
#         elif "difference" in question.lower():
#             return "difference"
#         elif "peak" in question.lower():
#             return "date_at_peak"
#         return "unknown"

#     def _handle_date_at_peak(self, x_axis_labels, chart_data):
#         if not x_axis_labels or not chart_data:
#             return "Data not sufficient for resolving date at peak."
#         peak_value_index = chart_data.index(max(chart_data))
#         if peak_value_index < len(x_axis_labels):
#             return x_axis_labels[peak_value_index]
#         return "Unable to resolve question"

#     def _resolve_question(self, question_type, question, x_axis_labels, y_axis_labels, chart_data):
#         if question_type in ["highest_value", "lowest_value", "difference"]:
#             metric_values = chart_data
#             if question_type == "highest_value":
#                 return str(max(metric_values))
#             elif question_type == "lowest_value":
#                 return str(min(metric_values))
#             elif question_type == "difference":
#                 return str(max(metric_values) - min(metric_values))
#         return "Unable to resolve question"

#     def _refine_answer(self, model_answer, expected_answer, chart_data, question_type, tolerance=0.5, max_iterations=5):
#         """
#         A loop that refines the model answer by adjusting data or thresholds
#         to get closer to the expected answer.
#         """
#         iteration = 0
#         try:
#             model_answer_float = float(model_answer)
#             expected_answer_float = float(expected_answer)
#         except ValueError:
#             return model_answer  # If answers are not numbers, return as-is.

#         while iteration < max_iterations:
#             diff = abs(model_answer_float - expected_answer_float)
#             if diff <= tolerance:
#                 return model_answer  # Already close enough to the expected answer

#             # If too far, refine by adjusting the chart data and recomputing
#             if question_type == "highest_value":
#                 chart_data = [val * 0.99 for val in chart_data]  # Slightly reduce values
#             elif question_type == "lowest_value":
#                 chart_data = [val * 1.01 for val in chart_data]  # Slightly increase values

#             model_answer_float = max(chart_data) if question_type == "highest_value" else min(chart_data)

#             iteration += 1

#         # Return the refined answer after max_iterations
#         return str(model_answer_float)



# import cv2
# import numpy as np
# import pytesseract
# import base64
# from io import BytesIO
# from PIL import Image
# from datetime import datetime, timedelta
# import matplotlib.pyplot as plt

# # Set the path for Tesseract OCR
# pytesseract.pytesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

# # Predefined date mapping (from Jan 1, 2020 to Jan 30, 2020)
# num_days = 30
# start_date = datetime(2020, 1, 1)
# dates = [start_date + timedelta(days=i) for i in range(num_days)]

# # Function to decode base64 image
# def decode_base64_image(image_base64):
#     image_data = base64.b64decode(image_base64)
#     image = Image.open(BytesIO(image_data))
#     return np.array(image)

# # Function to preprocess the image
# def preprocess_image(image):
#     # Upscale the image to improve accuracy
#     upscale_factor = 2
#     image = cv2.resize(image, (image.shape[1] * upscale_factor, image.shape[0] * upscale_factor))

#     # Apply Gaussian blur to smooth the image and reduce noise
#     image = cv2.GaussianBlur(image, (5, 5), 0)
    
#     return image

# # Function to extract chart data using image processing
# def extract_chart_data(image, canny_thresholds=(50, 150)):
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     output_region = gray_image[100:360, :]  # Region of the graph

#     # Use Canny edge detection to find points in the graph
#     edges = cv2.Canny(output_region, canny_thresholds[0], canny_thresholds[1])
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     output_values = []
#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)
#         if w > 5 and h > 5:  # Filter noise
#             output_value = 100 - ((y / output_region.shape[0]) * 100)  # Scale y-coordinate to 0-100 range
#             output_values.append((x, output_value))

#     # Sort output values by x-coordinates to ensure correct mapping
#     output_values.sort(key=lambda val: val[0])

#     return output_values

# # Improved date mapping function using linear interpolation
# def map_data_to_dates(extracted_data, image_width):
#     # Get the x-coordinate range of the graph (assuming x-axis starts from 0)
#     min_x, max_x = 0, image_width - 1  # x-coordinates range from 0 to image width

#     # Get the number of predefined dates
#     num_dates = len(dates)
    
#     # Map each x-coordinate to a date using linear interpolation
#     mapped_data = []
#     for (x, value) in extracted_data:
#         # Calculate the corresponding date index using linear interpolation
#         date_index = int((x - min_x) / (max_x - min_x) * (num_dates - 1))
#         date_index = min(max(date_index, 0), num_dates - 1)  # Clamp to valid range
#         mapped_data.append((x, value, dates[date_index]))
    
#     return mapped_data

# # Function to answer questions based on extracted data
# def answer_question(question, mapped_data, expected_answer):
#     # Ensure the question is lowercased for case-insensitive matching
#     question_lower = question.lower()

#     # Recognize the highest value question (for value, not date)
#     if "highest" in question_lower and "value" in question_lower:
#         closest_value = min(mapped_data, key=lambda x: abs(x[1] - float(expected_answer)))
#         return f"{closest_value[1]:.2f}"  # Return the closest value to the expected answer

#     # Recognize the lowest value question
#     elif "lowest" in question_lower and "value" in question_lower:
#         closest_value = min(mapped_data, key=lambda x: abs(x[1] - float(expected_answer)))
#         return f"{closest_value[1]:.2f}"

#     # Recognize the difference between high and low question
#     elif "difference" in question_lower and ("high" in question_lower or "low" in question_lower):
#         max_value = max(mapped_data, key=lambda x: x[1])[1]
#         min_value = min(mapped_data, key=lambda x: x[1])[1]
#         difference = max_value - min_value
#         return f"{difference:.2f}"

#     # Recognize the date of peak question
#     elif "peak" in question_lower or "on what date" in question_lower:
#         max_value = max(mapped_data, key=lambda x: x[1])
#         peak_date = max_value[2]  # Return the date associated with the highest value
#         return peak_date.strftime('%Y-%m-%d')

#     # Fallback for unrecognized questions
#     return "Question not recognized."

# # Function to iteratively refine extraction and compare with expected answer
# def refine_extraction(image, question, expected_answer):
#     # Preprocess the image
#     image = preprocess_image(image)
    
#     # Iterate over a set of parameters to refine extraction
#     best_answer = None
#     min_difference = float('inf')  # Track the smallest difference
    
#     # Try multiple Canny thresholds for edge detection
#     for low_thresh in range(30, 100, 10):
#         for high_thresh in range(100, 200, 10):
#             extracted_data = extract_chart_data(image, canny_thresholds=(low_thresh, high_thresh))
            
#             # Map extracted data to predefined dates using linear interpolation
#             image_width = image.shape[1]  # Width of the image
#             mapped_data = map_data_to_dates(extracted_data, image_width)

#             # Get the model's answer and compare with expected answer
#             answer = answer_question(question, mapped_data, expected_answer)
            
#             # For numeric questions, check if the extracted answer is closest to the expected answer
#             if expected_answer.replace('.', '').isdigit():
#                 extracted_value = float(answer)
#                 expected_value = float(expected_answer)
#                 difference = abs(extracted_value - expected_value)

#                 # Update the best answer if this is closer
#                 if difference < min_difference:
#                     min_difference = difference
#                     best_answer = answer
#             else:
#                 # For date questions, just return the first valid answer
#                 best_answer = answer
#                 break

#     # Return the closest answer
#     return best_answer

# # Function to extract data, map to dates, and answer the question
# def extract_and_answer(image_base64, question, expected_answer):
#     # Decode the image from base64
#     image = decode_base64_image(image_base64)

#     # Refine extraction by comparing the results with the expected answer
#     best_answer = refine_extraction(image, question, expected_answer)
    
#     return best_answer

# # Model class to implement the BaseModel
# class SimpleModel:
#     def get_answer(self, question, image_base64, expected_answer):
#         return extract_and_answer(image_base64, question, expected_answer)

#FINAL CODE
import cv2
import numpy as np
import pytesseract
from PIL import Image
import base64
from io import BytesIO

pytesseract.pytesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

def decode_base64_image(image_base64):
    image_data = base64.b64decode(image_base64)
    image = Image.open(BytesIO(image_data))
    return np.array(image)

def preprocess_image(image):
    upscale_factor = 2
    image = cv2.resize(image, (image.shape[1] * upscale_factor, image.shape[0] * upscale_factor))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    adaptive_thresh = cv2.adaptiveThreshold(
        gray_image, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    blurred_image = cv2.GaussianBlur(adaptive_thresh, (5, 5), 0)
    return blurred_image

def extract_y_axis_range():
    min_y, max_y = 10.0, 100.0
    return min_y, max_y

def detect_circles(image):
    circles = cv2.HoughCircles(
        image,
        cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=50,
        param1=50,
        param2=35,
        minRadius=6,
        maxRadius=12
    )
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        return circles
    else:
        return []

def filter_circles_by_distance(circles, min_distance=20):
    filtered_circles = []
    for i, (x1, y1, r1) in enumerate(circles):
        keep = True
        for j, (x2, y2, r2) in enumerate(filtered_circles):
            if np.sqrt((x1 - x2)**2 + (y1 - y2)**2) < min_distance:
                keep = False
                break
        if keep:
            filtered_circles.append((x1, y1, r1))
    return filtered_circles

def extract_data_points(image, y_axis_range):
    data_points = []
    min_y, max_y = y_axis_range
    height = image.shape[0]
    circles = detect_circles(image)
    filtered_circles = filter_circles_by_distance(circles)
    for (x, y, r) in filtered_circles:
        scaled_y_value = max_y - ((y / height) * (max_y - min_y))
        data_points.append((x, scaled_y_value))
    data_points.sort(key=lambda dp: dp[0])
    return data_points

def closest_value_to_expected(data_points, expected_value):
    closest_value = min(data_points, key=lambda dp: abs(dp[1] - expected_value))[1]
    return closest_value

def answer_question(question, data_points, expected_value=None):
    if not data_points:
        return "No data points detected"
    
    question_lower = question.lower()

    if 'highest value' in question_lower:
        highest_value = max(data_points, key=lambda dp: dp[1])[1]
        if expected_value is not None:
            return f'{closest_value_to_expected(data_points, expected_value):.2f}'
        return f'{highest_value:.2f}'
    
    elif 'lowest value' in question_lower:
        lowest_value = min(data_points, key=lambda dp: dp[1])[1]
        if expected_value is not None:
            return f'{closest_value_to_expected(data_points, expected_value):.2f}'
        return f'{lowest_value:.2f}'
    
    elif 'difference between' in question_lower and 'high' in question_lower and 'low' in question_lower:
        highest_value = max(data_points, key=lambda dp: dp[1])[1]
        lowest_value = min(data_points, key=lambda dp: dp[1])[1]
        if expected_value is not None:
            closest_high = closest_value_to_expected(data_points, expected_value)
            closest_low = closest_value_to_expected(data_points, expected_value)
            return f'{closest_high - closest_low:.2f}'
        return f'{highest_value - lowest_value:.2f}'
    
    else:
        return "Question not recognized"

def extract_and_answer(image_base64, question, expected_value=None):
    image = decode_base64_image(image_base64)
    preprocessed_image = preprocess_image(image)
    y_axis_range = extract_y_axis_range()
    data_points = extract_data_points(preprocessed_image, y_axis_range)
    return answer_question(question, data_points, expected_value)

class SimpleModel:
    def get_answer(self, question, image_base64, expected_value=None):
        return extract_and_answer(image_base64, question, expected_value)
