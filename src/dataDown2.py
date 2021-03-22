import requests 

def download_url(url, save_path, chunk_size=32768):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

print("Positive Data downloading...")
download_url("https://vyn-opensource-ai-datasets.s3-eu-west-1.amazonaws.com/ai-height-detection/data3/pos2.zip","./data/data2/positive.zip")
print("Negative Data downloading...")
download_url("https://vyn-opensource-ai-datasets.s3-eu-west-1.amazonaws.com/ai-height-detection/data3/neg2.zip","./data/data2/negative.zip")
print("Done!")

print("Positive Cross Data downloading...")
download_url("https://vyn-opensource-ai-datasets.s3-eu-west-1.amazonaws.com/ai-height-detection/cross-validation/cross_valid_positive.zip","./data/data2/cross_positive.zip")
print("Negative Cross Data downloading...")
download_url("https://vyn-opensource-ai-datasets.s3-eu-west-1.amazonaws.com/ai-height-detection/cross-validation/cross_valid_negative.zip","./data/data2/cross_negative.zip")
print("Done!")
