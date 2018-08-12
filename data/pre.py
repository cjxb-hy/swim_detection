import os  

pre_dict='./infer'
total_pre = os.listdir(pre_dict)  

num=len(total_pre)  
list1=list(range(num))  
fpre = open('./infer.txt', 'w')  

for i in list1:  
    name='./infer/'+total_pre[i]+'\n'  
    fpre.write(name)
  
fpre.close()
