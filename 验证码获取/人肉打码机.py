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
def login():
    global cookie
    header={'Host': 'xk.urp.seu.edu.cn',
                 'Connection': 'keep-alive',
                 'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
                 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.75 Safari/537.36',
                 'Origin': 'http://xk.urp.seu.edu.cn'
                 }
    postdic = collections.OrderedDict()#有序
    postdic[ 'userName']=''
    postdic[ 'password']=''
    postdic[ 'vercode']=''
    postdic[ 'x']='32'
    postdic[ 'y']='2'
    checkCode=str(root.edit_vercode.get())
    postdic['vercode'] = checkCode
    header.setdefault('Referer', 'http://xk.urp.seu.edu.cn/studentService/system/showLogin.action')
    data=parse.urlencode(postdic).encode('utf-8')
    req = request.Request('http://xk.urp.seu.edu.cn/studentService/system/login.action', data, headers=header)
    response = request.urlopen(req, timeout=12)
    text = response.read().decode('utf-8','ignore')
    if "学生服务" in text:
        return True;
    else:
        return False;
def huoqu():
    global cookie
    global image
    global im
    global root
    global num
    global headerdic
    global img_url
    if(login()==True):
        im.save(  check_filename_available(str(root.edit_vercode.get())+'.jpg') )
        num=num+1
        print("验证码",str(root.edit_vercode.get()),"样本数:",num)


    getCheckCode(img_url,headerdic)

    root.edit_vercode = tkinter.Entry(root)
    root.edit_vercode.grid(row=2, column=1, columnspan='2')
    # 读取验证码图片
    filename = 'verifycode.jpg'
    root.canvas = tkinter.Canvas(root)
    image_xianshi = Image.open(filename)
    image_xianshi = image_xianshi.resize((100, 50), Image.ANTIALIAS)
    img_bg = ImageTk.PhotoImage(image_xianshi)
    root.label = tkinter.Label(root, image=img_bg)
    root.label.image_xianshi = img_bg
    root.label.grid(row=3, column=1, columnspan='2')
    root.edit_vercode.bind('<Key-Return>', huoqu2)
    tkinter.Label(root, text="样本数"+str(num)).grid(row=5, column=0, padx=10, pady=10)
def huoqu2(event):
    global cookie
    global image
    global im
    global root
    global num
    global headerdic
    global img_url

    if(login()==True):
        im.save(  check_filename_available(str(root.edit_vercode.get())+'.jpg') )
        num=num+1
        print("验证码",str(root.edit_vercode.get()),"样本数:",num)

    getCheckCode(img_url,headerdic)

    root.edit_vercode = tkinter.Entry(root)
    root.edit_vercode.grid(row=2, column=1, columnspan='2')
    root.edit_vercode.focus()
    # 读取验证码图片
    filename = 'verifycode.jpg'
    root.canvas = tkinter.Canvas(root)
    image_xianshi = Image.open(filename)
    image_xianshi = image_xianshi.resize((100, 50), Image.ANTIALIAS)
    img_bg = ImageTk.PhotoImage(image_xianshi)
    root.label = tkinter.Label(root, image=img_bg)
    root.label.image_xianshi = img_bg
    root.label.grid(row=3, column=1, columnspan='2')
    root.edit_vercode.bind('<Key-Return>', huoqu2)
    tkinter.Label(root, text="样本数"+str(num)).grid(row=5, column=0, padx=10, pady=10)
    

if __name__ == '__main__':
    global cookie
    global image
    global im
    global root
    global num
    global headerdic
    global img_url
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
    
    img_url="http://xk.urp.seu.edu.cn/studentService/getCheckCode"
    getCheckCode(img_url,headerdic)

    root = root=tkinter.Tk()
    root.title("验证码打码")
    root.resizable(0,0)
    root.geometry('250x150')

    tkinter.Label(root, text="验证码").grid(row=2, column=0, padx=10, pady=10)
    root.edit_vercode = tkinter.Entry(root)
    root.edit_vercode.grid(row=2, column=1, columnspan='5')
    root.edit_vercode.focus()
    # 读取验证码图片
    filename = 'verifycode.jpg'
    root.canvas = tkinter.Canvas(root)
    image_xianshi = Image.open(filename)
    image_xianshi = image_xianshi.resize((100, 50), Image.ANTIALIAS)
    img_bg = ImageTk.PhotoImage(image_xianshi)
    root.label = tkinter.Label(root, image=img_bg)
    root.label.image_xianshi = img_bg
    root.label.grid(row=3, column=1, columnspan='2')
    root.edit_vercode.bind('<Key-Return>', huoqu2)
    tkinter.Label(root, text="样本数"+str(num)).grid(row=5, column=0, padx=10, pady=10)

    root.btn_submit = tkinter.Button(root, text='   提交   ', command= huoqu)
    root.btn_submit.grid(row=3, column=0, padx=10, pady=10)

    

    

    
    root.mainloop()  