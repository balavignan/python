file = open("D:\\python\\icp2\\abc.txt","r+")
wordcount={}
for word in file.read().split():
    if word.lower() not in wordcount:
        wordcount[word.lower()] = 1
    else:
        wordcount[word.lower()] += 1
print(wordcount)
file.close()
f = open("D:\\python\\icp2\\abc.txt", "w+")
for i in wordcount:
     f.write(i+" : "+str(wordcount[i])+"\n")
f.close()