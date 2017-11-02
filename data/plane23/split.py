import os
import shutil

img_root = '/home/alala/Projects/part_constellation_models_tensorflow/data/plane23/'
#img_root = '/Users/Alala/Projects/part_constellation_models_tensorflow/data/plane23'

root = "/home/alala/Projects/part_constellation_models_tensorflow/data/plane23/V8"
#root = '/Users/Alala/Projects/part_constellation_models_tensorflow/data/plane23/V8'

label = open('/home/alala/Projects/part_constellation_models_tensorflow/data/plane23/label.txt','r')

lines = label.readlines()

label_list = []
imgdir_list = []

for line in lines:
	label_id = line.split()[0]
	label_name = line.split()[1]
	label_list.append([label_id,label_name])

print label_list

for dirpath, dirs, files in os.walk(root):
	for dir in dirs:
		for label in label_list:
			if dir == label[1]:
				imgdir_list.append([os.path.join(root,dir),int(label[0]),label[1]])
				break

imgdir_list.sort(key=lambda x:x[1])


for imginfo in imgdir_list:
	imgpath = imginfo[0]
	for root, dirs, files in os.walk(imgpath):
		num = len(files)
		imginfo.append(num)
		i = 1
		for file in files:
			#formatenum = '%06d' % i
			#newname = os.path.join(imgpath,formatenum + ".jpg")
			#if not os.path.exists(newname):
			#	os.rename(os.path.join(imgpath,file),newname)
			i = i + 1

print imgdir_list

train = open('/home/alala/Projects/part_constellation_models_tensorflow/data/plane23/train.txt','w')
train.truncate()
test = open('/home/alala/Projects/part_constellation_models_tensorflow/data/plane23/test.txt','w')
test.truncate()

for imginfo in imgdir_list:
	imgpath = imginfo[0]
	class_num = imginfo[1]
	class_name = imginfo[2]
	pic_num = imginfo[3]

	# create dir
	train_class_dir = 'train/' + class_name
	test_class_dir = 'test/' + class_name

	if not os.path.exists(train_class_dir):
		os.mkdir(train_class_dir)

	if not os.path.exists(test_class_dir):
		os.mkdir(test_class_dir)

	for root, dirs, files in os.walk(imgpath):
		files.sort(key=lambda x: int(x[:-4]))
		for file in files:
			index = int(file[:-4])
			print dir
			print root
			if index % 5 == 0:
				line = img_root + train_class_dir + '/' + file + ' ' + str(class_num) + '\n'
				test.write(line)
				dst_dir = test_class_dir + '/' + file
				shutil.copyfile(os.path.join(imgpath,file), dst_dir)
			else:
				line = img_root + test_class_dir + '/' + file + ' ' + str(class_num) + '\n'
				train.write(line)
				dst_dir = train_class_dir + '/' + file
				shutil.copyfile(os.path.join(imgpath, file), dst_dir)

train.close()
test.close()

