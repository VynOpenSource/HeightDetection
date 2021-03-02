data = ./data/
models = ./models/
testing = ./testingImages/
src = ./src/

test2:
	@python $(src)testing2.py
	
test3:
	@python $(src)testing3.py

getData:
	@python $(src)dataDown.py
	@python $(src)unzipData.py

getModel2:
	@python $(src)modelDown2.py
	
getModel3:
	@python $(src)modelDown3.py

train2:
	@python $(src)2classmodel.py
	
train3:
	@python $(src)3classmodel.py