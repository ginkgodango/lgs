import sys
import ftplib
import os
from ftplib import FTP
import datetime as dt
ftp = FTP("ftp.msci.com")
ftp.login('wnvyzpng', 'hxcksyyx')


def downloadFiles(path, destination):
    """
    path & destination are str of the form "/dir/folder/something/"
    path should be the abs path to the root FOLDER of the file tree to download
    """
    count = 1
    try:
        ftp.cwd(path)
        # clone path to destination
        os.chdir(destination)
        os.mkdir(destination[0:len(destination)-1]+path)
        print(destination[0:len(destination)-1]+path+" built")
    except OSError:
        #folder already exists at destination
        pass
    except ftplib.error_perm:
        # invalid entry (ensure input form: "/dir/folder/something/")
        print("error: could not change to "+path)
        sys.exit("ending session")

    # list children:
    filelist = ftp.nlst()

    for file in filelist:
        try:
            # this will check if file is folder:
            ftp.cwd(path+file+"/")
            # if so, explore it:
            downloadFiles(path+file+"/", destination)
        except ftplib.error_perm:
            # not a folder with accessible content
            # download & return
            os.chdir(destination)
            # possibly need a permission exception catch:
            ftp.retrbinary("RETR " + file, open(file, "wb").write)
            with open(destination + '/Record.txt', 'a') as tf:
                tf.write(str(count) + ',' + str(file) + ',' + 'downloaded' + ',' + str(dt.datetime.now()) + '\n')
            print(count, file + " downloaded", str(dt.datetime.now()))
        count += 1
    return


source = "/download/history/"
dest = "F:/download/history/"

with open(dest + '/Record.txt', 'w') as tf:
    tf.write('Count, Filepath, Outcome, Timestamp\n')

downloadFiles(source, dest)

