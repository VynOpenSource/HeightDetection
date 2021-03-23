import requests 

def download_url(url, save_path, chunk_size=32768):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

print("Positive Data downloading...")
download_url("https://vyn-opensource-ai-datasets.s3-eu-west-1.amazonaws.com/ai-height-detection/training-data/train_positive.zip","./data/data3/positive.zip")
print("Negative Data downloading...")
download_url("https://vyn-opensource-ai-datasets.s3-eu-west-1.amazonaws.com/ai-height-detection/training-data/train_negative.zip","./data/data3/negative.zip")
print("Hard Data downloading...")
download_url("https://vyn-opensource-ai-datasets.s3-eu-west-1.amazonaws.com/ai-height-detection/training-data/train_hard.zip","./data/data3/hard.zip")
print("Done!")

print("Positive Cross Data downloading...")
download_url("https://vyn-opensource-ai-datasets.s3-eu-west-1.amazonaws.com/ai-height-detection/cross-validation/cross_valid_positive.zip","./data/data3/cross_positive.zip")
print("Negative Cross Data downloading...")
download_url("https://vyn-opensource-ai-datasets.s3-eu-west-1.amazonaws.com/ai-height-detection/cross-validation/cross_valid_negative.zip","./data/data3/cross_negative.zip")
print("Hard Cross Data downloading...")
download_url("https://vyn-opensource-ai-datasets.s3-eu-west-1.amazonaws.com/ai-height-detection/cross-validation/cross_valid_hard.zip","./data/data3/cross_hard.zip")
print("Done!")
