import zipfile
zip_ref = zipfile.ZipFile("./data/data2/negative.zip", 'r')
zip_ref.extractall("./data/data2/train")
zip_ref = zipfile.ZipFile("./data/data2/positive.zip", 'r')
zip_ref.extractall("./data/data2/train")

import zipfile
zip_ref = zipfile.ZipFile("./data/data2/cross_negative.zip", 'r')
zip_ref.extractall("./data/data2/cross_val")
zip_ref = zipfile.ZipFile("./data/data2/cross_positive.zip", 'r')
zip_ref.extractall("./data/data2/cross_val")
zip_ref.close()