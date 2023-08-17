import os
import re
import csv

def searchDirectory(cwd,searchParam,searchResults):
    dirs = os.listdir(cwd)
    for dir in dirs:
        fullpath = os.path.join(cwd,dir)
        if os.path.isdir(fullpath):
            searchDirectory(fullpath,searchParam,searchResults)
        if re.search(searchParam,fullpath):
            searchResults.append(fullpath)

searchParam = r'(medicoesGerais.dat)'

root = input("Input: root folder (no / at end)\n")
searchResults = []
searchDirectory(root,searchParam,searchResults)    

allHeaders = []

for file in searchResults:
    with open(file, encoding='ANSI') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        row = next(csv_reader)
    for head in row:
        if not head in allHeaders:
            allHeaders.append(head)

print("dict = {\n\t: \"" + "\",\n\t: \"".join(allHeaders)+"\"\n}")
input("Tarefa finalizada")