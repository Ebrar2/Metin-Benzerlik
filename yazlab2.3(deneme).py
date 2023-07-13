#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tkinter as tk
import string 
import nltk
import numpy as np
from numpy.linalg import norm
import tensorflow_hub as hub
import networkx as nx
import spacy
import PIL
from math import log10
import matplotlib.pyplot as plt
from numpy import dot
from numpy.linalg import norm
from nltk.stem import PorterStemmer
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from nltk.corpus import stopwords
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from rouge import Rouge
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image, ImageTk



global dokumanCumle
def dosyayiAc():
    global filename 
    filename=tk.filedialog.askopenfilename(initialdir="/",
                                 title="Select a File",
                                 filetypes=(("Text files",
                                            "*.txt*"),
                                            ("all files",
                                             "*.*")))
    
    label_file_explorer.configure(text="File Opened: " + filename)
   
 
def dosyaOku():
    file=filename.replace("/",'\\\\')
    fp=open(file,"r",encoding="utf8") 
    dokumanCumle=fp.read()
    cumleler=(dokumanCumle.split('.'))
    global baslik
    baslik=cumleler[0][:cumleler[0].index("\n")]
    cumleler[0]=cumleler[0][cumleler[0].index("\n")+1:]
    print("BASLIK:"+baslik)
    print("CUMLE0:"+cumleler[0])
    fp.close()
    sonCumleler=[]
    for i in cumleler:
        if(len(i.strip())!=0):
            sonCumleler.append(i)
    return sonCumleler
def kosinusBenzerlikBul(a,b):
    return round(dot(a, b)/(norm(a)*norm(b)),2)

def benzerlikToplam():
    for i in range(len(G)):
        toplam=0
        for j in range(len(G)):
            if(i!=j):
                bul=G.get_edge_data(i,j)['weight']
                toplam+=bul
        G.nodes[i]['benzerlikT']=toplam
 
 
    
def grafOlustur():
    global G
    G = nx.Graph()
    global cumleler
    cumleler=dosyaOku()
    for i in range(len(cumleler)):
        print(str(i)+"->"+cumleler[i])
        G.add_node(i,eski=cumleler[i]+".",vektor=[],yeniMetin=cumleler[i], color="blue",benzerlikS=0,benzerlikT=0.0,skor=0.0)
    degerler=[]
    E=[]
    onIslemAdımlarıUygula(cumleler)
    for i in range(len(cumleler)):
        for j in range(i+1,len(cumleler)):
            degerler.clear()
            degerler.append(i)
            degerler.append(j)
            degerler.append(kosinusBenzerlikBul((G.nodes[i]['vektor']), (G.nodes[j]['vektor'])))
            E.append(tuple(degerler))
    G.add_weighted_edges_from(E)
    benzerlikToplam()
    colors = [node[1]['color'] for node in G.nodes(data=True)]
    plt.figure(figsize=(70, 70))
    pos=nx.spring_layout(G, k=0.5, iterations=100, scale=2.0)
    #positonSet(G)
    nx.draw(G,pos,node_size=1000,font_size=20,node_color=colors, with_labels=True, font_weight='bold')
    edge_weight = nx.get_edge_attributes(G,'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_weight)
    plt.savefig("graf.png")
    plt.show()

    
def onIslemAdımlarıUygula(cumleler):
    module_url = "C:////Users//ebrar//Downloads//UNIVERSAL-SENTENCE-ENCODER" 
    word_embeed = hub.load(module_url)
    for i in range(len(cumleler)):
        kelime=G.nodes[i]['yeniMetin']
        kelime=kelime.lower()
        kelime=kelime.translate(str.maketrans('', '', string.punctuation))
        stop_words = set(stopwords.words('english'))
        kelime = word_tokenize(kelime)
        stop_word_delete = [w for w in kelime if not w.lower() in stop_words]
        stop_word_delete = []
        for x in kelime:
            if x not in stop_words:
                stop_word_delete.append(x)
        kelime=stop_word_delete
        ps = PorterStemmer()
        sno = nltk.stem.SnowballStemmer('english')
        stemmer_word=[]
        for x in kelime:
            stemmer_word.append(sno.stem(x))
        kelime=stemmer_word
        G.nodes[i]['yeniMetin']=kelime
        G.nodes[i]['vektor']=word_embeed(kelime)[0]
  

  
def grafBenzerlikOraniOlustur():
    benzerlikOrani=0
    newGraph = nx.Graph()
    try:
        benzerlikOrani = float(inputGet())
        pass
    except ValueError:
        messagebox.showwarning("Hata", "Hatalı Değer Girişi")
        return
    for i in range(len(G)):
        say=0
        for j in range(len(G)):
            if(i!=j):
                if(G.get_edge_data(i,j)['weight']>=benzerlikOrani):
                    say+=1
        newGraph.add_node(str(i)+"\n"+"x:"+str(say), color="blue")
        G.nodes[i]['benzerlikS']=say
    E=[]
    degerler=[]
    for i in range(len(G)):
        for j in range(i+1,len(G)):
            degerler.clear()
            str1=str(i)+"\n"+"x:"+str(G.nodes[i]['benzerlikS'])
            str2=str(j)+"\n"+"x:"+str(G.nodes[j]['benzerlikS'])
            degerler.append(str1)
            degerler.append(str2)
            degerler.append(G.get_edge_data(i,j)['weight'])
            E.append(tuple(degerler))
    newGraph.add_weighted_edges_from(E)
    for i in range(len(G)):
        for j in range(i+1,len(G)):
            str1=str(i)+"\n"+"x:"+str(G.nodes[i]['benzerlikS'])
            str2=str(j)+"\n"+"x:"+str(G.nodes[j]['benzerlikS'])
            if(G.get_edge_data(i,j)['weight']>=benzerlikOrani):
                newGraph.edges[str1, str2]['color']='r'
            else:
                 newGraph.edges[str1, str2]['color']='b'        
    edges = newGraph.edges()
    colors = [newGraph[u][v]['color'] for u,v in edges]
    #pos=nx.circular_layout(newGraph)
    plt.figure(figsize=(70, 70))
    pos=nx.spring_layout(newGraph, k=0.5, iterations=100, scale=2.0)
    nx.draw(newGraph, pos,node_size=1500,font_size=10,edge_color=colors, with_labels=True, font_weight='bold')
    edge_weight = nx.get_edge_attributes(newGraph,'weight')
    nx.draw_networkx_edge_labels(newGraph, pos, edge_labels = edge_weight)
    plt.savefig("grafBenzerlik"+str(benzerlikOrani)+".png")
    plt.show()
    grafSkorOlustur()
        
def inputGet():
    return inputtxt.get(1.0, "end-1c")

def inputskorGet():
    return inputskor.get(1.0, "end-1c")
def getP1(sentence):
    nlp = spacy.load('en_core_web_sm')
    sayac=0
    prWords = nlp(sentence)
    for ent in prWords.ents:
        sayac+=1
    return sayac
    
def getP2(sentence):
    yenisentence=word_tokenize(sentence)
    sayac=0
    for i in yenisentence:
        if(i.isnumeric()==True):
            sayac+=1
    return sayac

def getP3(i):
    return G.nodes[i]['benzerlikS']

def getP4(sentence):
    baslikParcala=word_tokenize(baslik)
    sentenceParcala=word_tokenize(sentence)
    sayac=0
    for i in baslikParcala:
        for j in sentenceParcala:
            if(i.lower()==j.lower()):
                sayac+=1
    return sayac

def getTumCumleler():
    tmCumleler=[]
    for i in range(len(G)):
        tmCumleler.extend(G.nodes[i]['yeniMetin'])
    return tmCumleler
def getTumCumlelerOnIslemUygulanmis():
    birlestir=set(G.nodes[0]['yeniMetin'])
    for i in range(2,len(G)):
        sentence=set(G.nodes[i]['yeniMetin'])
        birlestir=birlestir.union(sentence)
    return list(birlestir)



def bulTFIDF(kelime,cumle):
    TF = cumle.count(kelime) / len(cumle)
    say = 0
    for i in cumle:
        if i == kelime:
            say += 1
            break
    DF = len(cumle) / say
    IDF = log10(DF)
    return TF*IDF

def temaKelimeleriBul(essizKelimeler,tumMetinler):
    tumTFIDF=dict()
    for i in range(len(essizKelimeler)):
        tumTFIDF[essizKelimeler[i]]=bulTFIDF(essizKelimeler[i],tumMetinler)
    toplamKelimeSay=len(tumMetinler)
    temaSayisi=(int)(toplamKelimeSay/10)
    yeniSiralanmis=(sorted(tumTFIDF.items(), key=lambda x:x[1],reverse=True)).copy()
    temaKelimeler=[]
    for i in range(temaSayisi):
        yaz=yeniSiralanmis.pop(0)
        temaKelimeler.append(yaz[0])
    return temaKelimeler
    
def getP5(temaKelimeler,nodeKelimeler):
    toplamTemaSay=0
    for i in range(len(temaKelimeler)):
        kactaneVar=nodeKelimeler.count(temaKelimeler[i]) 
        toplamTemaSay+=kactaneVar
    return toplamTemaSay
    
def grafSkorOlustur():
    tumMetinler=getTumCumleler()
    essizKelimeler=getTumCumlelerOnIslemUygulanmis()
    temaKelimeler=temaKelimeleriBul(essizKelimeler,tumMetinler)
    toplamBaglantiSay=len(G)-1
    global tumSkorlar
    tumSkorlar=dict()
    global ortalamaSkor
    ortalamaSkor=0.0
    for i in range(len(G)):
        sentence=G.nodes[i]['eski']
        sentenceYeni=G.nodes[i]['yeniMetin']
        cumleUzunluk=len(sentenceYeni)
        print("---------------------------"+str(i)+"----------------------------------------")
        p1=getP1(sentence)/cumleUzunluk+1
        p2=getP2(sentence)/cumleUzunluk+1
        p3=getP3(i)/toplamBaglantiSay+1
        p4=getP4(sentence)/cumleUzunluk+1
        p5=getP5(temaKelimeler,sentenceYeni)/cumleUzunluk+1
        benzerT=(G.nodes[i]['benzerlikT']+1)
        skor=(p5*p1*p2*p3*p4)*benzerT
        #skor=p5+p1+p2+p3+p4+benzerT
        G.nodes[i]['skor']=skor
        tumSkorlar[i]=skor
        print("SKOR:"+str(skor))
        ortalamaSkor+=skor
        print("----------------------------------------------------------")
    ortalamaSkor=ortalamaSkor/(toplamBaglantiSay+1)
    print("ORTALAMA SKOR:"+str(ortalamaSkor))
    tumSkorlar=sorted(tumSkorlar.items(), key=lambda x:x[1],reverse=True)
    grafSkorGoster()
        
def grafSkorGoster():
    newGraph = nx.Graph()
    for i in range(len(G)):
        skor=round(G.nodes[i]['skor'],2)
        newGraph.add_node(str(i)+"\n"+"y:"+str(skor), color="blue")
    E=[]
    degerler=[]
    for i in range(len(G)):
        for j in range(i+1,len(G)):
            degerler.clear()
            str1=str(i)+"\n"+"y:"+str(round(G.nodes[i]['skor'],2))
            str2=str(j)+"\n"+"y:"+str(round(G.nodes[j]['skor'],2))
            degerler.append(str1)
            degerler.append(str2)
            degerler.append(G.get_edge_data(i,j)['weight'])
            E.append(tuple(degerler))
    newGraph.add_weighted_edges_from(E)
    #pos=nx.circular_layout(newGraph)
    plt.figure(figsize=(70, 70))
    pos=nx.spring_layout(newGraph, k=0.5, iterations=100, scale=2.0)
    nx.draw(newGraph, pos,node_size=1500,font_size=10, with_labels=True, font_weight='bold')
    edge_weight = nx.get_edge_attributes(newGraph,'weight')
    nx.draw_networkx_edge_labels(newGraph, pos, edge_labels = edge_weight)
    plt.savefig("grafSkor.png")
    plt.show()
    
def ozetle():
    try:
        girilenSkor = float(inputskorGet())
        pass
    except ValueError:
        messagebox.showwarning("Hata", "Hatalı Değer Girişi")
        return
    global ozetCumle
    ozetCumle=""
    tmskorlar=dict()
    tmskorlar=tumSkorlar.copy()
    print(tumSkorlar)
    print(ortalamaSkor)
    for i in range((int)(len(G))):
        cumle=tmskorlar.pop(0)
        if(cumle[1]<ortalamaSkor):
            break
        if(cumle[1]>=girilenSkor and cumle[1]>=ortalamaSkor):
            ozetCumle=ozetCumle+" "+G.nodes[cumle[0]]['eski'].strip()
        else:
            break
    print(ozetCumle)
    ozetGoster()


    
def ozetGoster():
    pencereOzet=Tk()
    pencereOzet.title("Ozet")
    pencereOzet.geometry("600x600")
    label_file_ozet= Text(pencereOzet,width=100,height=100)
    label_file_ozet.insert(tk.END, ozetCumle)
    label_file_ozet.grid(column=1,row=1)

def rougeSkoruHesapla(verilenOzetMetin):
    rouge = Rouge()
    scores = rouge.get_scores(ozetCumle.lower(), verilenOzetMetin, avg=True)
    return scores
    

def dosyayiAcSkor():
    global filename2
    filename2=tk.filedialog.askopenfilename(initialdir="/",
                                 title="Select a File",
                                 filetypes=(("Text files",
                                            "*.txt*"),
                                            ("all files",
                                             "*.*")))
    
    label_file_rougeAl.configure(text="File Opened: " + filename2)
    

def rougeskorGoster():
    file=filename2.replace("/",'\\\\')
    fp=open(file,"r",encoding="utf8") 
    dokumanCumle=fp.read()
    rougeSkor=rougeSkoruHesapla(dokumanCumle.lower())
    bir=round(rougeSkor['rouge-1']['f'],2)
    iki=round(rougeSkor['rouge-2']['f'],2)
    uc=round(rougeSkor['rouge-l']['f'],2)
    strmetin="rouge-1:"+str(bir)+"\n"+"rouge-2:"+str(iki)+"\n"+"rouge-l:"+str(uc)
    label_file_rouge= Text(pencere,width=20,height=5)
    label_file_rouge.insert(tk.END, strmetin)
    label_file_rouge.grid(column=2,row=1)

    
    
    
    
pencere = Tk()
pencere.title("yazlab2.3")
pencere.geometry("600x600")
pencere.config(background="purple")


    
label_file_explorer = tk.Label(pencere,
                            text="Bir Dosya Seçiniz",
                            width=50, height=3,
                            fg="gray")

button_explore = tk.Button(pencere,
                        text="Dosyalara Göz At",
                        command=dosyayiAc)

button_exit = tk.Button(pencere,
                     text="Çıkış",
                     command=grafOlustur)
label_file_benzerlik=tk.Label(pencere,
                            text="Cümle Benzerlik Thresholdu Giriniz",
                            width=50, height=3,
                            fg="black")

inputtxt = tk.Text(pencere,
                   height = 3,
                   width = 20)
button_bezerlikGrafOlustur = tk.Button(pencere,
                     text="Tamam",
                     command=grafBenzerlikOraniOlustur)

label_file_skor=tk.Label(pencere,
                            text="Cümle Skor Thresholdu  Giriniz",
                            width=50, height=3,
                            fg="black")

inputskor = tk.Text(pencere,
                   height = 3,
                   width = 20)
button_skorGrafOlustur = tk.Button(pencere,
                     text="Özet Oluştur",
                     command=ozetle)
label_file_rougeAl = tk.Label(pencere,
                            text="Bir Dosya Seçiniz",
                            width=50, height=3,
                            fg="gray")
button_rougeAl = tk.Button(pencere,
                        text="Dosyalara Göz At",
                        command=dosyayiAcSkor)
button_hesapla = tk.Button(pencere,width=20,
                     text="ROUGE Hesapla",
                     command=rougeskorGoster)


# Grid metodu araçların yerleşimi
label_file_explorer.grid(column=0, row=1)
button_explore.grid(column=0, row=2)
button_exit.grid(column=0, row=3)
label_file_benzerlik.grid(column=0,row=4)
inputtxt.grid(column=0,row=5)
button_bezerlikGrafOlustur.grid(column=0,row=6)
label_file_skor.grid(column=0,row=7)
inputskor.grid(column=0,row=8)
button_skorGrafOlustur.grid(column=0,row=9)
label_file_rougeAl.grid(column=0,row=14)
button_rougeAl.grid(column=0,row=15)
button_hesapla.grid(column=0,row=16)
pencere.mainloop()


# In[ ]:




