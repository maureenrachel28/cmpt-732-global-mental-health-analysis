#!/usr/bin/env python
# coding: utf-8

# In[10]:


from PyPDF2 import PdfReader
from os import listdir
from os.path import isfile, join
import pandas as pd
import re
from tabula import read_pdf
from tabulate import tabulate


# In[31]:


folder = r'C:\Users\ASUS\Desktop\Naveen\SFU\Sem_1\Lab\Project\Excel Dataset Final'


# In[26]:


# Import the required Module
import tabula
# Read a PDF File
df = tabula.read_pdf("Afghanistan.pdf", pages='all')[2]
# convert PDF into CSV
# tabula.convert_into("IPLmatch.pdf", "iplmatch.csv", output_format="csv", pages='all')
df


# In[18]:


onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]
onlyfiles


# In[19]:


df = pd.DataFrame(columns=['Country','Population','MH Expenditure','Disabled','Mortality','Articles',
                          'MH Research Output','MH Research Total'])
df


# In[45]:


for path in onlyfiles:
    reader = PdfReader(folder+'\\'+path)
    number_of_pages = len(reader.pages)
    for curr_page in range(number_of_pages):
        page = reader.pages[curr_page]
        text = page.extract_text()
        if curr_page == 0:
#             x = read_pdf(folder+'\\' + pages=1,relative_area = True,
#              area = [8 , 40 , 15, 50], guess = False,)
            print(x[0])
            #             population = int((text.split('\n', 1)[0]).replace(" ", ""))            
#             mh_expenditure =re.findall("\d+\.\d+",(text.split('\n', 1)[1]))
            
#             print('$$',population, mh_expenditure)
#             row = pd.Series([population, 'Franc', 3.3, 'CS', 'Paris',1,1,1], index = df.columns)
#             df = pd.concat([df, row.to_frame().T])
            
        print(text,'\n--------------------------------------')
        
    print('\n','_____________________________________________________________________________________________')


# In[44]:


for path in onlyfiles:
    reader = PdfReader(folder+'\\'+path)
    number_of_pages = len(reader.pages)
    pop = read_pdf(folder+'\\' +path, pages=1,relative_area = True,
        area = [8 , 40 , 15, 50], guess = False)
    row = pd.Series(['country', pop[0].values[0][0], 3.3, 'CS', 'Paris',1,1,1],index = df.columns)
    df = pd.concat([df, row.to_frame().T])
    print(pop[0].values[0][0], type(pop[0].values[0]))        
    print('\n','_____________________________________________________________________________________________')


# In[7]:


df


# In[8]:


df['Population'] = df['Population'].astype(str)
df.dtypes


# In[9]:


df['Population'].replace(to_replace='[^0-9]+', value='',inplace=True,regex=True)


# In[10]:


df


# In[11]:


x = read_pdf(folder+'\\' +'afg.pdf', pages=1,relative_area = True,
             area = [8 , 40 , 15, 50], guess = False,)
x[0]


# In[12]:


type(x[0])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




