import os
import xmltodict
def dict2arr(dict):
    l=['0','0','0','0']
    l[0]=dict['xmin']
    l[1] = dict['ymin']
    l[2]=dict['xmax']
    l[3]=dict['ymax']
    return l

def print2txt(lsts,fnm):
    save_path = 'D:/Custom_Dataset_GT/annotation/'
    if fnm.endswith('.jpg'):
        fnme=fnm.replace('.jpg','_GT.txt')
    else:
        fnme=fnm.replace('.png','_GT.txt')
    completeName = os.path.join(save_path, fnme)
    with open(completeName, 'w+') as file:
        for lst in lsts:
            if lst[-1]=='"Blank"':
                if lst==lsts[-1]:
                    file.write("")
                else:
                    file.write('\n')
            else:
                line = " ".join(lst)
                file.write(line+'\n')
    file.close()
  
def pprocdata(dir):
    sum=0
    for filename in os.scandir(dir):
        if filename.is_file():
            print(filename.path)
        # PARSE XML FILE
            with open(filename.path) as xmlfile:
                xml = xmltodict.parse(xmlfile.read())
            lsts=[]
            # sum=0
            for object in xml['annotation']['object']:
                li=[]
                boundingbox=object['bndbox']
                bbx=dict2arr(boundingbox)
                txt=object['name']
                text=f'"{txt}"'
                li.extend(['0', '0', '0', '0', '0'])
                li.extend(bbx)
                li.append(text)
                lsts.append(li)
                # print(i,li)
                # sum+=len(lsts)
            sum+=len(lsts)
            print(f'No. of lines : {len(lsts)}')
            print(sum)
            print2txt(lsts,xml['annotation']['filename'])

pprocdata('D:/Custom_Dataset_GT/Annotations/')
