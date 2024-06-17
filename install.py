import os
import re
import wget
from os import listdir
from os.path import isdir, join
import shutil

cwd=os.getcwd()
only_dirs = [dir for dir in listdir(cwd) if isdir(join(cwd, dir))]
bsp_url = "https://github.com/OpenNuvoton/M55M1BSP/archive/refs/tags/"

for dir in only_dirs:
    if re.search("M55M1BSP-", dir):
        bsp_name = dir
        bsp_version = bsp_name.replace("M55M1BSP-", "")
        break

if bsp_version == None:
    exit()

bsp_url = bsp_url + "V" + bsp_version + ".zip"
print("Start download from {}".format(bsp_url))
bsp_zip_file = wget.download(bsp_url)
print("")
print("download done")
print("unzip bsp zip file: {}".format(bsp_zip_file))
shutil.unpack_archive(bsp_zip_file)
print("remove zip file")
os.remove(bsp_zip_file)
print("copy patch files to BSP")
shutil.copytree('patch', bsp_name, dirs_exist_ok=True)
print("install done")
