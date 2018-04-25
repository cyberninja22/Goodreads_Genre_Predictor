import pickle
a = ['test value 1','test value 2 ','test value 3 ']

file_name = 'testfile'
fileObject = open(file_name,'wb')

pickle.dump(a,fileObject)

file_Object =open(file_name,'r')

b = pickle.load(fileObject)

print(a==b)