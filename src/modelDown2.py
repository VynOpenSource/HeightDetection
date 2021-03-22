import requests 

def download_url(url, save_path, chunk_size=32768):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

print("Model2 starts downloading...")
download_url("https://vyn-opensource-ai-datasets.s3-eu-west-1.amazonaws.com/ai-height-detection/models/mymodel_2class.h5","./models/myModel_2class.h5")
print("Model2 successfully downloaded in 'models' folder!")