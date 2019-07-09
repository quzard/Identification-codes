#coding:utf-8
import http.cookiejar
from urllib import request
from urllib import parse

import collections
import matplotlib.pyplot as plt
import numpy as np
import tkinter
import os
from PIL import Image, ImageTk
from bs4 import BeautifulSoup
from multiprocessing.dummy import Pool as ThreadPool
import time
    
def getUrlResponse(url,head) :
    global cookie
    url = str(url)
    req = request.Request(url)
    for eachhead in head.keys():
        req.add_header(eachhead,head[eachhead])
        
    resp = request.urlopen(req)  
    return resp
def check_filename_available(filename):
    n=[0]
    def check_meta(file_name):
        file_name_new=file_name
        if os.path.isfile(file_name):
            file_name_new=file_name[:file_name.rfind('.')]+'_'+str(n[0])+file_name[file_name.rfind('.'):]
            n[0]+=1
        if os.path.isfile(file_name_new):
            file_name_new=check_meta(file_name)
        return file_name_new
    return_name=check_meta(filename)
    return return_name
def getCheckCode(url,headerdic):
    global cookie
    global image
    global im
    cookie = http.cookiejar.LWPCookieJar()
    opener =request.build_opener(request.HTTPCookieProcessor(cookie), request.HTTPHandler)
    request.install_opener(opener)
    image = request.urlopen(url, timeout=8)

    f = open('verifycode.jpg', 'wb')
    f.write(image.read())
    f.close()
    im = Image.open('verifycode.jpg')
    im=im.resize((160, 60))
    im.save('verifycode.jpg')
def main(checkCode):
    global cookie
    global yanzhen
    global final
    global now_time
    global last_time
    global start_time
    if(final==0):
        header={'Host': 'xk.urp.seu.edu.cn',
                     'Connection': 'keep-alive',
                     'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
                     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.75 Safari/537.36',
                     'Origin': 'http://xk.urp.seu.edu.cn'
                     }
        header.setdefault('Referer', 'http://xk.urp.seu.edu.cn/studentService/system/showLogin.action')
        postdic = collections.OrderedDict()#有序
        postdic[ 'userName']=''
        postdic[ 'password']=''
        postdic[ 'vercode']=''
        postdic[ 'x']='32'
        postdic[ 'y']='2'
        postdic['vercode'] = checkCode
        
        data=parse.urlencode(postdic).encode('utf-8')
        req = request.Request('http://xk.urp.seu.edu.cn/studentService/system/login.action', data, headers=header)
        response = request.urlopen(req, timeout=12)
        text = response.read().decode('utf-8','ignore')
        if "学生服务" in text and final==0:
            yanzhen=checkCode
            final=1
        else :
            print(checkCode,"样本数:",num,"final",final,"上次耗费时间:",str(now_time),"总耗费时间：",str(time.time()-start_time),end='\r', flush=True)

def login():
    global cookie
    global yanzhen
    global final
    checkCode=[]
    for i in range(10000):
        if(i<10):
            checkCode.append('000'+str(i))
        elif(i<100):
            checkCode.append('00'+str(i))
        elif(i<1000):
            checkCode.append('0'+str(i))
        elif(i<10000):
            checkCode.append(''+str(i))
    pool = ThreadPool(16)
    pool.map(main, checkCode)
    pool.close()
    pool.join()
    if(final==1):
        return True

def begin1():
    global cookie
    global yanzhen
    global image
    global im
    global root
    global num
    global headerdic
    global img_url
    global final
    global now_time
    global last_time
    global start_time
    while True:
        final=0
        if(login()==True):
            im.save(  check_filename_available(str(yanzhen)+'.jpg') )
            num=num+1
            now_time=time.time()-last_time
        getCheckCode(img_url,headerdic)
        last_time=time.time()
        

if __name__ == '__main__':
    global cookie
    global yanzhen
    global image
    global im
    global root
    global num
    global headerdic
    global img_url
    global now_time
    global last_time
    global start_time
    now_time=0
    last_time=0
    num=0
    TEXT=[]
    files = os.listdir('.')#获得当前 硬盘目录中的所有文件  
    for i in files:#逐个文件遍历  
        if( os.path.isfile(i)):# 判断当前是一个文件夹'''   
            if(os.path.splitext(i)[1]=='.jpg'and os.path.splitext(i)[0]!='verifycode'):# 当前不是文件夹 获得当前的文件的扩展名  
                num=num+1;
                TEXT.append(os.path.splitext(i)[0][:4])
    headerdic={'Host': 'xk.urp.seu.edu.cn',
				'Connection': 'keep-alive',
				'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
				'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.75 Safari/537.36',
				'Origin': 'http://xk.urp.seu.edu.cn'
				}
    start_time = time.time()
    last_time=start_time
    img_url="http://xk.urp.seu.edu.cn/studentService/getCheckCode"
    getCheckCode(img_url,headerdic)

    begin1()
    


    

    

    

    
    