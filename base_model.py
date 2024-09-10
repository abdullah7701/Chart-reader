from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def get_answer(self, question, image_base64):
        pass

# import base64
# from PIL import Image
# from io import BytesIO
# from task import generate_task  # Ensure you have the correct import

# def show_base64_image(base64_str):
#     """
#     Decodes a base64 string and displays the corresponding image.
#     """
#     image_data = base64.b64decode(base64_str)
#     image = Image.open(BytesIO(image_data))
#     image.show()

# # Call generate_task to get the base64 string
# question, image_base64, selected_answer = generate_task()

# # Display the chart image
# show_base64_image(image_base64)
