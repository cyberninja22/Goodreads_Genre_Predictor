import os , sys 

fd = open('foo.txt', 'wb' )

os.write(fd ,'This is a test')

os.close(fd)

print("The work has been done successfully")