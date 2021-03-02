import zipfile
zip_ref = zipfile.ZipFile("./data/negative.zip", 'r')
zip_ref.extractall("./data/train")
zip_ref = zipfile.ZipFile("./data/positive.zip", 'r')
zip_ref.extractall("./data/train")
zip_ref = zipfile.ZipFile("./data/hard.zip", 'r')
zip_ref.extractall("./data/train")

import zipfile
zip_ref = zipfile.ZipFile("./data/cross_negative.zip", 'r')
zip_ref.extractall("./data/cross_val")
zip_ref = zipfile.ZipFile("./data/cross_positive.zip", 'r')
zip_ref.extractall("./data/cross_val")
zip_ref = zipfile.ZipFile("./data/cross_hard.zip", 'r')
zip_ref.extractall("./data/cross_val")
zip_ref.close()