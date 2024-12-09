import requests

url = "http://localhost:9001/api/image"
file_path = "/api/test/test_images/photo_2022-11-07_14-21-03.jpg"

with open(file_path, "rb") as image_file:
    response = requests.post(url, files={"image": image_file})

print(response.json())
