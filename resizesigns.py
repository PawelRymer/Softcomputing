#!/usr/bin/env python

from PIL import Image
import os
from math import fabs
from numpy import random

''' UWAGA!!! Nie mozna tworzyc katalogu do zapisu obrazow tam gdzie wykonujemy program ani w podkatalogach '''
''' Trzeba to robic w folderze wyzej albo jakimkolwiek innym bo przeszukujemy wszystkie podkatalogi '''
# do ustawienia: 
format = ".png"     # format foteczek wejsciowych
size = 28           # rozmiar docelowy w pix         
path_out_validate = "E:/Studia/Softcomputing/resized_out_foto/validation_set_characters/" # sciezka do zapisu zbioru uczacego
path_out_train = "E:/Studia/Softcomputing/resized_out_foto/characters/" # sciezka do zapisu zbioru walidujacego
path_in = os.getcwd()  # sciezka do folderu z foteczkami do przerobienia          
                       # (tutaj folder w ktorym wykonujemy program)
                       # (przeszukuje katalog glowny i podkatalogi!)
def get_images():
    '''Zwraca liste ze sciezkami do obrobionych plikow.'''
    outfiles = []
    i = 1
    number = 1
    file = open('tooSmallImages.txt', 'a')
    charFolderHeader = ''
    trainMax = 6000
    testMax = 1000
    for (dirpath, dirnames, filenames) in os.walk(path_in):
        for filename in [f for f in filenames if format in f]:
            #output = 'dir' + str(i) + '_' + filename
            foldersNames = dirpath.split('\\')
            if charFolderHeader != foldersNames[4]:
                totalTrain = 0
                totalTest = 0
                number = 1
            charFolderHeader = foldersNames[4]
            index = random.randint(1, 6)
            output = 'char_' + foldersNames[4] + '_' + str(number) + format
            number += 1
            # .convert('L') konwertuje do skali szarosci
            # usun jesli chcesz kolorowe
            img = Image.open(dirpath + "/" + filename).convert('L')
    
    
            height = int(img.size[1])
            width = int(img.size[0])
            difference = int(fabs((height - width) / 2))

            # path = dirpath + "\\" + filename

            if height < size or width < size:
                print "obrazek mniejszy niz wymagany wymiar - " +filename
                # os.remove(path)
                file.write(filename+'\n')
                continue
            if height >= width:
                area = (0, difference, width, difference + width)
                new_img = img.crop(area)
                new_img = new_img.resize((size, size))
                if index != 5 and totalTrain < trainMax:
                    if not os.path.exists(path_out_train + charFolderHeader):
                        os.makedirs(path_out_train + charFolderHeader)
                    new_img.save(path_out_train + charFolderHeader + '/' + output)
                    totalTrain = totalTrain+1
                    outfiles.append(path_out_train + charFolderHeader + '/' + output)
                elif index == 5 and totalTest < testMax:
                    if not os.path.exists(path_out_validate + charFolderHeader):
                        os.makedirs(path_out_validate + charFolderHeader)
                    new_img.save(path_out_validate + charFolderHeader + '/' + output)
                    totalTest = totalTest + 1
                    outfiles.append(path_out_train + charFolderHeader + '/' + output)
                elif totalTrain >= trainMax and totalTest >= testMax:
                    break
            else:
                area = (difference, 0, difference + height, height)
                new_img = img.crop(area)
                new_img = new_img.resize((size, size))
                if index != 5 and totalTrain < trainMax:
                    if not os.path.exists(path_out_train + charFolderHeader):
                        os.makedirs(path_out_train + charFolderHeader)
                    new_img.save(path_out_train + charFolderHeader + '/' + output)
                    totalTrain = totalTrain + 1
                    outfiles.append(path_out_train + charFolderHeader + '/' + output)
                elif index == 5 and totalTest < testMax:
                    if not os.path.exists(path_out_validate + charFolderHeader):
                        os.makedirs(path_out_validate + charFolderHeader)
                    new_img.save(path_out_validate + charFolderHeader + '/' + output)
                    totalTest = totalTest + 1
                    outfiles.append(path_out_train + charFolderHeader + '/' + output)
                elif totalTrain == trainMax and totalTest == testMax:
                    break
        i = i + 1
    file.close()
    return outfiles
get_images()

