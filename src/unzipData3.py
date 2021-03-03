import zipfile
zip_ref = zipfile.ZipFile("./data/data3/negative.zip", 'r')
zip_ref.extractall("./data/data3/train")
zip_ref = zipfile.ZipFile("./data/data3/positive.zip", 'r')
zip_ref.extractall("./data/data3/train")
zip_ref = zipfile.ZipFile("./data/data3/hard.zip", 'r')
zip_ref.extractall("./data/data3/train")

import zipfile
zip_ref = zipfile.ZipFile("./data/data3/cross_negative.zip", 'r')
zip_ref.extractall("./data/data3/cross_val")
zip_ref = zipfile.ZipFile("./data/data3/cross_positive.zip", 'r')
zip_ref.extractall("./data/data3/cross_val")
zip_ref = zipfile.ZipFile("./data/data3/cross_hard.zip", 'r')
zip_ref.extractall("./data/data3/cross_val")
zip_ref.close()