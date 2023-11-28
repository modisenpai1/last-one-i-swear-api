import requests

url = 'http://127.0.0.1:5000/upload_image'  # Replace with your Flask endpoint URL
file_path = 'H:/DeepFashion2DataSet/test/image/000120.jpg'  # Replace with the path to your image file

files = {'file': open(file_path, 'rb')}  # File to be sent in the request

try:
    response = requests.post(url, files=files)

    if response.status_code == 200:
        print("File uploaded successfully")
        print(response.json())  # Print the response from the server
    else:
        print(f"Request failed with status code: {response.status_code}")
except requests.RequestException as e:
    print(f"Request exception: {e}")

