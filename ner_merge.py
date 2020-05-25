
from __future__ import unicode_literals, print_function
import spacy
import pandas as pd
import cv2
import os,glob
from pdf2image import convert_from_path
from os import listdir,makedirs
from os.path import isfile,join
import pika 
from os.path import abspath
import json
from spacy import *
from tkinter import filedialog
from tkinter import *
root = Tk()
root.withdraw()



path1=abspath('/')

path= filedialog.askdirectory(initialdir = "/",title = "Select Invoice Directory") 


dstpath = abspath('./gray')
print(dstpath) # Destination Folder

try:
    makedirs(dstpath)
except:
    print ("Directory already exist, images will be written in same folder")
# Folder won't used
files = [f for f in listdir(path) if isfile(join(path,f))] 
for image in files:
    if image.endswith('.jpg'):
        
        try:
            img = cv2.imread(os.path.join(path,image))
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            dstPath = join(dstpath,image)
            cv2.imwrite(dstPath,gray)
        except:
            print ("{} is not converted".format(image))
    else:
        pages=convert_from_path(path+image,500)
        for page in pages:
            page.save(dstpath+image+'.jpg', 'JPEG')
            
for fil in glob.glob("*.jpg"):
    try:
        image = cv2.imread(fil) 
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert to greyscale
        cv2.imwrite(os.path.join(dstpath,fil),gray_image)
    except:
        pages=convert_from_path(path+images,500)
        for page in pages:
            page.save(dstpath+images+'.jpg', 'JPEG')
            
path=abspath('./gray/')   
dstpath = abspath('./denoised/')  
dstpath1 = abspath('./csv/') 
dstpath2 = abspath('./json/') 
print(path)
try:
    makedirs(dstpath)
    makedirs(dstpath1)
    makedirs(dstpath2)
except:
    print ("Directory already exist, images will be written in same folder")
# Folder won't used
files = [f for f in listdir(path) if isfile(join(path,f))] 
for image in files:
    try:
        img = cv2.imread(os.path.join(path,image))
        dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
        dstPath = join(dstpath,image)
        cv2.imwrite(dstPath,dst)
    except:
        print ("{} is not converted".format(image))
for fil in glob.glob("*.jpg"):
    try:
        image = cv2.imread(fil) 
        den_image = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21) 
        cv2.imwrite(os.path.join(dstpath,fil),den_image)
    except:
        print('{} is not converted')
        
dstpath = abspath('./denoised/')  
print(dstpath)          
import pytesseract
pytesseract.pytesseract.tesseract_cmd=filedialog.askopenfilename(initialdir = "/",title = "Select Tesseract File", filetypes =[('Files', '*.exe')])    

for j in os.listdir(dstpath+'/'):
    if j.endswith('.jpg'):
        try:
            from PIL import Image
        except ImportError:
            import Image
        import pytesseract
        text = pytesseract.image_to_string(Image.open(dstpath+'/' + j))  # We'll use Pillow's Image class to open the image and pytesseract to detect the string in the image
    with open('./'+j+'.txt', 'a') as a_writer:
        a_writer.write(text)

from spacy.util import set_data_path
spacy.util.set_data_path('C:/Users/Gokul P/Anaconda3/Lib/site-packages/spacy/data')
output_dir1=filedialog.askdirectory(initialdir = "/",title = "Select Model Directory")
print(output_dir1)
nlp2 = spacy.load(output_dir1)
for j in os.listdir('./'):
    if j.endswith('.txt'):
        with open (j,'r') as f:
            test_text = f.read()
            #print(j)
#     test_text = '''Hima Santhosh Ernakulam Singing  
        
        # '''
            label=[]
            text=[]
            doc = nlp2(test_text)
            print("Entities in '%s'" % test_text)
            for ent in doc.ents:
                # label = ent.label_
                # text = ent.text
                # a=print(ent.label_, ent.text)
                label.append(ent.label_)
                text.append( ent.text)
                # print(label)
                d = {'label':label,'text':text}
                df = pd.DataFrame(d,index=None)
                print(df)
                # ent.save(path1+'new/'+j+'.jpg', 'JPEG')
                df.to_csv('./'+j+'.csv')
                
for root,dirs,files in os.walk('./'):
    for x in files:
        if x.endswith(".csv"):
            print(x)
            f=pd.read_csv(x, delimiter=',')
            f.drop(['Unnamed: 0'],axis=1, inplace=True, index=None)
            f.to_csv('./'+"csv/"+x,index=None)
            
for j in os.listdir('./'+'csv/'):
    if j.endswith('.csv'):
        print(j)
        f=pd.read_csv(j, delimiter=',')
        a=f.groupby('label').agg({'text':lambda x :' '.join(map(str, x))})
        a.to_csv('./'+'csv1/'+j+'.csv')

def convert_to_json(fname):
    result = []
    rec = {}
    with open(fname) as f:
        for l in f:
            if not l.strip() or l.startswith('label'):
                continue

            if l.startswith(','):
                result.append(rec)
                rec = {}
            else:
                k, v = l.strip().split(',')
                if k.strip():
                    try:
                        rec[k] = int(v)
                    except:
                        rec[k] = v.strip('"')
                else:
                    rec['Comments'] += v.strip('"')
    result.append(rec)
    return result
directory = './'+'csv1/'
print(directory)

for root,dirs,files in os.walk(directory):
    print('****************************************************************')
    
    for x in files:
        if x.endswith(".csv"):
        # print(x)
            data=convert_to_json('./'+'csv1/'+x)
            
            with open('./'+'json/'+x+'.json', 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

import pika    
pathz='./'+"json/"
import json
for j in os.listdir(pathz):
    if j.endswith('.json'):
        print(j)
        with open(pathz+j) as f:
            d = json.load(f)
            print(d)
            connection = pika.BlockingConnection(
                    pika.ConnectionParameters(host='localhost'))
            channel = connection.channel()
            channel.queue_declare(queue='hello')
            message =d
            channel.basic_publish(exchange='', routing_key='hello',body=json.dumps(message))
            print(" [x] JSON "+j+"Recieved")
            connection.close()


    
                
                
                
                

            
           
    
# filepath='/content/a.json'
# with open(filepath,'w')as jsonFile:
#   jsonFile.write(json.dump(data, filepath))








    
            
 


