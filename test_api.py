import requests


URL = "http://127.0.0.1:8000/predict"

data = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}

response = requests.post(URL, json=data)

assert response.status_code == 200

result = response.json()

assert "prediction" in result
assert result["prediction"] == 0

print("API test passed")
print("Response:", result)