data = ./data/
models = ./models/
testing = ./testingImages/
src = ./src/

test2:
	@python $(src)testing2.py
	
test3:
	@python $(src)testing3.py

getData2:
	@python $(src)dataDown2.py
	@python $(src)unzipData2.py
	
getData3:
	@python $(src)dataDown3.py
	@python $(src)unzipData3.py

getModel2:
	@python $(src)modelDown2.py
	
getModel3:
	@python $(src)modelDown3.py

train2:
	@python $(src)2classmodel.py
	
train3:
	@python $(src)3classmodel.py