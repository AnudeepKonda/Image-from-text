import urllib.request
import re
import os


file_path = os.path.dirname(os.path.abspath(__file__)) + "\\pubfig_imgs"
with open("dev_urls.txt") as img_file:
    count = 0
    print("Starting to fetch imgs")
    count = 0
    for line in img_file:
        arr = re.split(' |\t', line)
        url = arr[3]
        extension = url[url.index('.')+1:]
        img_name = file_path + "\\" + arr[0] + "-" + arr[1] + "-" + arr[2] + extension
        print("Fetching image ", url, ' ', count+1)
        try:
            urllib.request.urlretrieve(url, img_name)
        except Exception as e:
            print (str(e))
        count += 1