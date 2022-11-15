ff = open('caffe_result_fp.txt','w')  #打开一个文件，可写模式
with open('new_caffe_result_fp.txt','r') as f:  #打开一个文件只读模式
    line = f.readlines()
    i = 0
    for line_list in line:
        #line_new =line_list.replace('./mydata/testing_data/','')  #将换行符替换为空('')
        #b = str(label) #主要是这一步 将之前列表数据转为str才能加入列表
        #line_new = line_new + label + '\n'
        line_new =line_list.replace('.jpg','')
        i += 1
        print(line_new)
        ff.write(line_new) #写入一个新文件中
