# import base64
# import io
# import random
# from datetime import datetime, timedelta
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from PIL import Image

# def generate_task():
#     # Color scheme for different metrics
#     plot_colors = [
#         "#FF6F71",  # Light Crimson
#         "#FFA07A",  # Light Salmon
#         "#FFD700",  # Light Goldenrod
#         "#9370DB",  # Light Indigo
#         "#FF9F80",  # Light Coral
#         "#AFEEEE",  # Light Turquoise
#         "#E6A8D7",  # Light Orchid
#         "#D2B48C",  # Light Sienna
#         "#20B2AA",  # Light Teal
#         "#8470FF",  # Light Slate Blue
#         "#E5E5FF",  # Light Periwinkle
#         "#ADFF2F",  # Light Chartreuse
#         "#B0E0E6",  # Light Cadet Blue
#         "#FF7F50",  # Light Tomato
#         "#9ACD32",  # Light Olive Drab
#         "#FF69B4",  # Light Medium Violet Red
#         "#FFDAB9",  # Light Peach Puff
#         "#98FB98",  # Light Lawn Green
#         "#F4A460",  # Light Rosy Brown
#         "#E0FFFF"  # Light Pale Turquoise
#     ]
#     bg_color = [
#         "#8B0000",  # Dark Crimson
#         "#E9967A",  # Dark Salmon
#         "#B8860B",  # Dark Goldenrod
#         "#2F4F4F",  # Dark Indigo
#         "#CD5B45",  # Dark Coral
#         "#008B8B",  # Dark Turquoise
#         "#9932CC",  # Dark Orchid
#         "#8B4513",  # Dark Sienna
#         "#006666",  # Dark Teal
#         "#483D8B",  # Dark Slate Blue
#         "#6666FF",  # Dark Periwinkle
#         "#556B2F",  # Dark Chartreuse
#         "#4682B4",  # Dark Cadet Blue
#         "#CD3700",  # Dark Tomato
#         "#556B2F",  # Dark Olive Drab
#         "#8B008B",  # Dark Medium Violet Red
#         "#CD853F",  # Dark Peach Puff
#         "#228B22",  # Dark Lawn Green
#         "#8B5F65",  # Dark Rosy Brown
#         "#5F9EA0"  # Dark Pale Turquoise
#     ]
#     labels_color = "#000000"  # Black for labels

#     # Terms for y-axis
#     y_terms = ['Efficiency', 'Productivity', 'Performance', 'Output', 'Speed', 'Quality', 'Rate', 'Capacity', 'Level', 'Intensity', 'Strength', 'Power']

#     # Generate data for each term over 30 days
#     num_days = 30
#     start_date = datetime(2020, 1, 1)
#     dates = [start_date + timedelta(days=i) for i in range(num_days)]
#     data = np.random.uniform(low=10, high=100, size=num_days)
#     df = pd.DataFrame({
#         'Date': dates,
#         'Output': data
#     })

#     plt.figure(figsize=(10, 6))
#     plt.plot(df['Date'], df['Output'], color='blue', marker='o', linestyle='-')

#     # Set title and labels with custom colors
#     plt.title('Output Over Time', color=labels_color)
#     plt.xlabel('Date', color=labels_color)
#     plt.ylabel('Values', color=labels_color)

#     # Customize y-axis to have more granularity
#     max_val = np.ceil(data.max() / 10) * 10  # Round to nearest 10
#     min_val = np.floor(data.min() / 10) * 10  # Round to nearest 10
#     plt.yticks(np.arange(min_val, max_val + 10, 5))  # Increase granularity by setting ticks every 5 units

#     plt.grid(True, which='both', linestyle='--', linewidth=0.5)
#     plt.gca().set_ylim([min_val, max_val])  # Set limits to rounded values
#     plt.legend(['Output'])

#     buf = io.BytesIO()
#     plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)  # High DPI for clearer text
#     buf.seek(0)

#     image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
#     buf.close()
#     plt.close()

#     # Generate questions and answers based on the data
#     questions = [
#         ('What is the highest value of Output?', f'{data.max():.2f}'),
#         ('What is the lowest value of Output?', f'{data.min():.2f}'),
#         ('What is the difference between the high and low of Output?', f'{(data.max() - data.min()):.2f}')
#     ]

#     # Add a date-related question, but it won't be included in the selection
#     date_question = ('On what date is Output at its peak?', str(df.loc[data.argmax(), 'Date'].date()))
    
#     # Randomly select one of the questions and its answer (date question not included)
#     selected_question, selected_answer = random.choice(questions)

#     return selected_question, image_base64, selected_answer

# if __name__ == '__main__':
#     question, image_base64, selected_answer = generate_task()
#     print(question)
#     print(selected_answer)
#     image = Image.open(io.BytesIO(base64.b64decode(image_base64)))
#     image.show()



import base64
import io
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

def generate_task_simple():
    light_colors = [
        "#FFD700", 
        "#20B2AA", 
        "#9370DB", 
        "#FF6F71", 
        "#ADFF2F", 
        "#FF69B4", 
        "#98FB98", 
        "#E6E6FA", 
        "#F08080", 
        "#FFE4B5", 
        "#E0FFFF", 
        "#FF6347", 
        "#FFE4E1", 
        "#B0C4DE", 
        "#D3D3D3", 
        "#FFB6C1", 
        "#FAFAD2", 
        "#F0E68C", 
        "#ADD8E6", "#FFA07A", "#AFEEEE", "#FFDAB9", "#EEE8AA", "#98FB98", "#D8BFD8", "#FFFAF0", "#FFEBCD", 
        "#FFEFD5", "#FFC0CB", "#FFDAB9", "#FFDEAD", "#FFF0F5", "#FFF5EE", "#FFFFE0", "#FFFACD", "#FAEBD7", 
        "#FFF8DC", "#F5DEB3", "#EEDD82", "#FAFAD2", "#D2B48C", "#E0FFFF", "#BDB76B", "#F4A460", "#FF69B4", 
        "#CD5C5C", "#E6E6FA", "#7B68EE", "#8A2BE2", "#5F9EA0", "#FF7F50", "#FFD700", "#EE82EE", "#DA70D6"
    ]
    
    dark_colors = ["#8B0000", "#228B22", "#9932CC", "#483D8B", "#CD3700", "#5F9EA0", "#FF4500", "#B22222", "#8A2BE2"]

    def serialize_image(image: Image.Image) -> str:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        img_str = base64.b64encode(img_bytes).decode('utf-8')
        return img_str

    def generate_evenly_spread_dates(start_date, num_days):
        spread_days = num_days * 2
        dates = [start_date + timedelta(days=int(spread_days * i / (num_days - 1))) for i in range(num_days)]
        return dates

    num_days = np.random.randint(15, 20)
    start_year = np.random.randint(1990, 2020)
    start_date = datetime(start_year, 1, 1)
    dates = generate_evenly_spread_dates(start_date, num_days)

    random_low = 10
    random_high = 100

    data = {'Date': dates, 'Value': np.random.uniform(low=random_low, high=random_high, size=num_days)}
    df = pd.DataFrame(data)
    df = df.sort_values(by='Date')

    plt.figure(figsize=(8, 4))
    plt.gca().set_facecolor(random.choice(light_colors))
    plt.plot(df['Date'], df['Value'], marker='o', color=random.choice(dark_colors), label='Value')

    plt.title('Value vs Date', color='black')
    plt.xlabel('Date', color='black')
    plt.ylabel('Value', color='black')
    plt.ylim(10, 100)
    plt.grid(True, color='gray')

    plt.xticks(rotation=45)
    plt.legend()

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
    buf.seek(0)

    image = Image.open(buf)
    image_base64 = serialize_image(image)
    plt.close()

    highest_value = df['Value'].max()
    lowest_value = df['Value'].min()
    diff_high_low = highest_value - lowest_value

    questions = [
        (f'What is the highest value?', f'{highest_value:.2f}'),
        (f'What is the lowest value?', f'{lowest_value:.2f}'),
        (f'What is the difference between the high and low values?', f'{diff_high_low:.2f}')
    ]
    
    question, selected_answer = random.choice(questions)

    return question, image_base64, selected_answer



####testing:
# import base64
# import io
# import random
# from datetime import datetime, timedelta
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from PIL import Image

# def generate_task():
#     # Color scheme for different metrics
#     plot_colors = ['#1f77b4', '#ff7f0e']  # Blue and Orange
#     bg_color = "#FFFFFF"  # White background
#     labels_color = "#000000"  # Black for labels

#     # Terms for y-axis
#     y_terms = ['Level', 'Intensity', 'Strength', 'Power', 'Output']

#     # Generate data for Output over 30 days
#     num_days = 30
#     start_date = datetime(2020, 1, 1)
#     dates = [start_date + timedelta(days=i) for i in range(num_days)]
#     data = np.random.uniform(low=10, high=100, size=num_days)
#     df = pd.DataFrame({
#         'Date': dates,
#         'Output': data
#     })

#     # Create the plot
#     plt.figure(figsize=(10, 6))
#     plt.plot(df['Date'], df['Output'], color='blue', marker='o', linestyle='-')

#     # Set title and labels with custom colors
#     plt.title('Output Over Time', color=labels_color)
#     plt.xlabel('Date', color=labels_color)
#     plt.ylabel('Values', color=labels_color)

#     # Customize y-axis to have more granularity
#     max_val = np.ceil(data.max() / 10) * 10  # Round to nearest 10
#     min_val = np.floor(data.min() / 10) * 10  # Round to nearest 10
#     plt.yticks(np.arange(min_val, max_val + 10, 5))  # Increase granularity by setting ticks every 5 units

#     plt.grid(True, which='both', linestyle='--', linewidth=0.5)
#     plt.gca().set_ylim([min_val, max_val])  # Set limits to rounded values
#     plt.legend(['Output'])

#     # Save the plot as a base64-encoded image
#     buf = io.BytesIO()
#     plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)  # High DPI for clearer text
#     buf.seek(0)

#     image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
#     buf.close()
#     plt.close()

#     # Generate questions and answers based on the data
#     questions = [
#         ('What is the highest value of Output?', f'{data.max():.2f}'),
#         ('What is the lowest value of Output?', f'{data.min():.2f}'),
#         ('What is the difference between the high and low of Output?', f'{(data.max() - data.min()):.2f}'),
#         ('On what date is Output at its peak?', str(df.loc[data.argmax(), 'Date'].date())),
#     ]

#     # Randomly select one of the questions and its answer
#     selected_question, selected_answer = random.choice(questions)

#     return selected_question, image_base64, selected_answer

# if __name__ == '__main__':
#     question, image_base64, selected_answer = generate_task()
#     print(question)
#     print(selected_answer)
#     image = Image.open(io.BytesIO(base64.b64decode(image_base64)))
#     image.show()
