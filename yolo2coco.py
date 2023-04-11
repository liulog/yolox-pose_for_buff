import os
import json
from PIL import Image

# 保存的路径
typename = "val"
coco_format_save_path='D:/Onedrive/桌面/buffdata/'                      #要生成的标准coco格式标签所在文件夹
yolo_format_classes_path='D:/Onedrive/桌面/buffdata/classes.txt'             #类别文件，一行一个类
yolo_format_annotation_path='D:/Onedrive/桌面/buffdata/'+typename+'/labels/'        #yolo格式标签所在文件夹
img_pathDir='D:/Onedrive/桌面/buffdata/'+typename+'/images/'                        #图片所在文件夹
name = os.path.join(coco_format_save_path,'buff_'+ typename+ '.json')

with open(yolo_format_classes_path,'r') as fr:                               #打开并读取类别文件
    lines1=fr.readlines()
# print(lines1)

# categories 从 1 -> num_classes
categories=[]                                                                 #存储类别的列表
for j,label in enumerate(lines1):
    label=label.strip()
    categories.append({ 'id':j+1,
                        'name': label,                    
                        'supercategory':'buff',
                        'keypoints':['right1','left1','left2','mid','right2'],  # 分别对应
                        'skeleton':[[0, 1],[1, 2],[2, 3],[3, 4],[4, 0],[2, 4]]
                       })                                                       #将类别信息添加到categories中

# print(categories)

# 准备json数据

write_json_context=dict()                                                      #写入.json文件的大字典
write_json_context['info']= {'description': ' RM buff datasets ', 'url': ' ', 'version': ' ', 'year': 2023, 'contributor': ' ', 'date_created': '2023-04-09'}
write_json_context['licenses']=[{'id':1,'name':None,'url':None}]

write_json_context['categories']=categories
write_json_context['images']=[]
write_json_context['annotations']=[]

#接下来的代码主要添加'images'和'annotations'的key值
imageFileList=os.listdir(img_pathDir)                                           #遍历该文件夹下的所有文件，并将所有文件名添加到列表中
for i,imageFile in enumerate(imageFileList):
    imagePath = os.path.join(img_pathDir,imageFile)                             #获取图片的绝对路径

    # 打开这个图片文件
    image = Image.open(imagePath)                                               #读取图片，然后获取图片的宽和高
    W, H = image.size

    img_context={}                                                              #使用一个字典存储该图片信息
    #img_name=os.path.basename(imagePath)                                       #返回path最后的文件名。如果path以/或\结尾，那么就会返回空值
    # file_name
    img_context['file_name']=imageFile
    
    # 宽高
    img_context['height']=H
    img_context['width']=W
    img_context['date_captured']='2023-04-09'
    
    # 统计id
    img_context['id']=i                                                         #该图片的id
    img_context['license']=1
    img_context['color_url']=' '
    img_context['flickr_url']=' '
    write_json_context['images'].append(img_context)                            #将该图片信息添加到'image'列表中

    # 从txt文件中读取
    txtFile=imageFile[:-3]+'txt'                                               #获取该图片获取的txt文件
    with open(os.path.join(yolo_format_annotation_path,txtFile),'r') as fr:
        lines=fr.readlines()                                                   #读取txt文件的每一行数据，lines2是一个列表，包含了一个图片的所有标注信息
    
    # 处理txt文件中的每一行
    for j,line in enumerate(lines):
        kps_dict = {}                                                          #将每一个bounding box信息存储在该字典中
        # line = line.strip().split()
        # print(line.strip().split(' '))

        class_id,x,y,w,h,x1,y1,x2,y2,x3,y3,x4,y4,x5,y5=line.strip().split(' ')                                          #获取每一个标注框的详细信息
        class_id,x,y,w,h,x1,y1,x2,y2,x3,y3,x4,y4,x5,y5 = int(class_id), float(x), float(y), float(w), float(h), \
            float(x1), float(y1), float(x2), float(y2), float(x3), float(y3), float(x4), float(y4), float(x5), float(y5)       #将字符串类型转为可计算的int和float类型

        # 检验框
        xmin=(x-w/2)*W                                                                    #坐标转换
        ymin=(y-h/2)*H
        xmax=(x+w/2)*W
        ymax=(y+h/2)*H
        
        # 宽高 目标恢复到 原图
        w=w*W
        h=h*H
        x1=W*x1
        x2=W*x2
        x3=W*x3
        x4=W*x4
        x5=W*x5
        y1=H*y1
        y2=H*y2
        y3=H*y3
        y4=H*y4
        y5=H*y5

        # 配置id
        # 一张图片最多出现5个大符的部分，只要不重复即可
        kps_dict['id']=i*10+j                                                            #bounding box的坐标信息
        kps_dict['image_id']=i
        # 类别数 + 1
        kps_dict['category_id']=class_id+1                                               #注意目标类别要加一
        kps_dict['iscrowd']=0
        kps_dict['num_keypoints']=5             # buff 5点格式，个数都是5
        kps_dict['keypoints']=[x1,y1,2,x2,y2,2,x3,y3,2,x4,y4,2,x5,y5,2]
        
        height,width=abs(ymax-ymin), abs(xmax-xmin)
        kps_dict['area']=height*width
        # 左上角坐标以及w h
        kps_dict['bbox']=[xmin,ymin,w,h]
        kps_dict['segmentation']=[[x1,y1,x2,y2,x3,y3,x4,y4,x5,y5]]
        write_json_context['annotations'].append(kps_dict)                               #将每一个由字典存储的bounding box信息添加到'annotations'列表中


with open(name,'w') as fw:                                                                #将字典信息写入.json文件中
    json.dump(write_json_context,fw,indent=2)
