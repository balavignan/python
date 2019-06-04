name = input("enter a string:")
char1 = input("enter char 1 to delete:")
char2 = input("enter char 2 to delete:")
newname = name.replace(char1,'')
newname1 = newname.replace(char2,'')
rname=""
for i in newname1:
    rname = i + rname
print("The reverse string is %s"%(rname))


