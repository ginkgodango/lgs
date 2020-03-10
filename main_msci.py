from ftplib import FTP
import os
import xml.etree.ElementTree as ET
import pandas as pd

ftp = FTP("ftp.msci.com")
ftp.login('wnvyzpng', 'hxcksyyx')
print(ftp.getwelcome())
print(ftp.pwd())
print(ftp.dir())


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

# total_size = get_total_size(ftp.pwd())

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

"""
lgs_msci_dict = {
    'AQR Delta': 'LG-HF01',
    'Winton': 'LG-HF02',
    'Attunga': 'LG-HF03',
    'GMO': 'LG-HF04',
    'Ardea': 'LG-HF05'
}
df_msci = df_jpm_main.copy()
df_msci['Year'] = df_msci['Date'].dt.year
df_msci['Month'] = df_msci['Date'].dt.month
df_msci = df_msci[['Manager', 'Year', 'Month', 'Return_JPM']]
df_msci = df_msci.pivot_table(index=['Manager', 'Year'], columns=['Month'], values=['Return_JPM'])
df_msci.columns = df_msci.columns.droplevel(0)
month_categories = [
    'Jan(%)', 'Feb(%)', 'Mar(%)', 'Apr(%)',
    'May(%)', 'Jun(%)', 'Jul(%)', 'Aug(%)',
    'Sep(%)', 'Oct(%)', 'Nov(%)', 'Dec(%)'
]
df_msci.columns = month_categories
df_msci[month_categories] = df_msci[month_categories]*100
df_msci = df_msci.reset_index(drop=False)
df_msci.insert(loc=2, column='ID Type', value='BARRAID')

msci_id = []
for i in range(0, len(df_msci)):
    if df_msci['Manager'][i] in lgs_msci_dict:
        msci_id.append(lgs_msci_dict[df_msci['Manager'][i]])
    else:
        msci_id.append(None)

df_msci['Id'] = msci_id
df_msci = df_msci[['Id', 'ID Type', 'Year'] + month_categories + ['Manager']]
df_msci.to_csv('U:/CIO/#Research/returns_for_msci_201910.csv', index=False)
"""