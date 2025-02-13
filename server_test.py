import requests
import json

# Define the URL of your Flask API
url = 'http://127.0.0.1:5000/predict'

# Define the data you want to send to the server
data = {
    'altitude': 2500,
    'average_temperature': 18,
    'average_humidity': 72
}

# Send a POST request with the data
response = requests.post(url, json=data)

# Print the response from the server
print(response.status_code)  # Status code (200 for success, 400 for error, etc.)
print(response.json())  # Response body (JSON)
