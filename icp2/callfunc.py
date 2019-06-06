def string_alternative():
    st = input("Enter the string : ")
    str1 = ""
    for i in range(len(st)):
        if(i%2==0):
            str1 = str1+st[i]
    print(str1)
    return 1
