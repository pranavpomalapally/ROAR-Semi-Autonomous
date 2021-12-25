import requests

r = requests.get("http://10.0.0.2:40001/save_world")
print(r.status_code)
print(r.text == "200")