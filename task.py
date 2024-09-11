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
