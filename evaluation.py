from datetime import datetime


def evaluate_processing_time(processing_time):
    timeout = 9.0
    max_score = 0.5
    if processing_time <= timeout/1.75:
        return max_score
    if processing_time <= timeout/1.25:
        return max_score/2
    if processing_time <= timeout:
        return max_score/5
    return -max_score/2


def evaluate_response(answer, expected_answer):
    def calculate_float_points(diff):
        if diff == 0:
            return 5
        elif diff <= 2:
            return 4.5
        elif diff <= 3:
            return 4.5
        elif diff <= 4:
            return 3.5
        elif diff <= 5:
            return 3.5
        elif diff <= 6:
            return 2.5
        elif diff <= 7:
            return 2.0
        elif diff <= 8:
            return 1.5
        elif diff <= 9:
            return 1.0
        else:
            return 0.5

    def calculate_date_points(diff):
        if diff == 0:
            return 5.0   
        elif diff <= 2:   
            return 4.5
        elif diff <= 3:
            return 4.5
        elif diff <= 4:
            return 3.5
        elif diff <= 6:
            return 3.5
        elif diff <= 7:
            return 2.5
        elif diff <= 10:
            return 2.0
        elif diff <= 15:
            return 1.5
        elif diff <= 20:
            return 1.0
        else:
            return 0.5

    try:
        # Check if the expected answer is numeric (float comparison)
        expected_answer = float(expected_answer)
        diff = abs(float(answer) - expected_answer)
        return calculate_float_points(diff)
    
    except ValueError:
        # Handle date comparison (strip whitespace and check format)
        expected_answer = expected_answer.strip()
        answer = answer.strip()

        # Direct string comparison first
        if expected_answer == answer:
            return 5.0
        
        # Convert to datetime and calculate date difference
        expected_answer_date = datetime.strptime(expected_answer, '%Y-%m-%d')
        answer_date = datetime.strptime(answer, '%Y-%m-%d')
        diff = abs((answer_date - expected_answer_date).days)

        return calculate_date_points(diff)



def evaluate(answer, expected_answer, processing_time):
    try:
        response_score = evaluate_response(answer, expected_answer)
        processing_score = evaluate_processing_time(processing_time)
        return (response_score + processing_score) / 5.5
    except:
        return -1.0