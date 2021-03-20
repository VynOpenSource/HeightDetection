import requests

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

print("Positive Data downloading...")
download_file_from_google_drive("14PdQEFWOG16n7f9CNOHqI1_IBwD7QS7n","./data/data3/positive.zip")
print("Negative Data downloading...")
download_file_from_google_drive("1g4MjbX_ifvJTghoGNBw85JGCSBuvySud","./data/data3/negative.zip")
print("Hard Data downloading...")
download_file_from_google_drive("1RK5WUCD2q_zWNvuE68ZT0tkN8RCFMYLZ","./data/data3/hard.zip")
print("Done!")

print("Positive Cross Data downloading...")
download_file_from_google_drive("1HQXAslvTXQAzvGJ6hKfSQBxZpvr0dQjH","./data/data3/cross_positive.zip")
print("Negative Cross Data downloading...")
download_file_from_google_drive("1MUtvQyI279UAhX15kWoc5f4RThtHaYcG","./data/data3/cross_negative.zip")
print("Hard Cross Data downloading...")
download_file_from_google_drive("1FPqJodFdYsC6JFlGbAdrsxGpyewq5FP2","./data/data3/cross_hard.zip")
print("Done!")
