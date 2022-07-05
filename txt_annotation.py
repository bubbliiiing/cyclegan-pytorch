import os

#-------------------------------------------------------#
#   datasets_path   指向数据集所在的路径
#-------------------------------------------------------#
datasets_path   = 'datasets'

if __name__ == "__main__":
    classes     = ['A', 'B']
    list_file   = open('train_lines.txt', 'w')

    types_name  = os.listdir(datasets_path)
    for type_name in types_name:
        if type_name not in classes:
            continue
        cls_id = classes.index(type_name)
        
        photos_path = os.path.join(datasets_path, type_name)
        photos_name = os.listdir(photos_path)
        for photo_name in photos_name:
            _, postfix = os.path.splitext(photo_name)
            if postfix not in ['.jpg', '.png', '.jpeg']:
                continue
            list_file.write(str(cls_id) + ";" + '%s'%(os.path.abspath(os.path.join(photos_path, photo_name))))
            list_file.write('\n')
    list_file.close()
