from ftplib import FTP
import os
import xml.etree.ElementTree as ET
import pandas as pd

ftp = FTP("ftp.msci.com")
ftp.login('wnvyzpng', 'hxcksyyx')
print(ftp.getwelcome())
print(ftp.pwd())

def get_total_size(ftp_dir):
    size = 0
    count = 0
    parent_dir = ftp.pwd() # get the current directory
    for filename in ftp.nlst(ftp_dir):
        # (don't forget to import os)
        path = os.path.join(parent_dir, filename) # keeps recursively track of the path
        try:
            ftp.cwd(path)
            size += get_total_size(path)
            ftp.cwd(parent_dir)
        except:
            ftp.voidcmd('TYPE I')
            size += ftp.size(path)
        count += 1
        print('Count: ', count, 'Size:', size, 'Filename: ', filename)
    return size

total_size = get_total_size(ftp.pwd())

# ftp.quit()

def load_msci(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()
    df = pd.DataFrame()
    for child0 in root:
        for child1 in child0:
            print(child1.attrib)
            df = df.append(child1.attrib, ignore_index=True)
    return df


# df_msci = load_msci('U:/CIO/#Research/MSCI/20200113core_dm_daily_d/20200113_20200113_CORE_DM_ALL_SECURITY_MAIN_DAILY_D.xml')
