import sys
import json
import pandas as pd

def predict_health_status(input_data):
    cpu = input_data.get('cpu', 0)
    memory = input_data.get('memory', 0)

    if cpu > 90 or memory > 90:
        status = "critical"
        score = 0.95
    elif cpu > 70 or memory > 70:
        status = "warning"
        score = 0.75
    else:
        status = "healthy"
        score = 0.99

    return {"status": status, "confidence": score}

if __name__ == "__main__":
    input_json = sys.argv[1]
    data = json.loads(input_json)
    result = predict_health_status(data)
    print(json.dumps(result))
