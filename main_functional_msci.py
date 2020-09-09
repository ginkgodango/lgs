import xml.etree.ElementTree as ET
import pandas as pd
import zipfile


def load_msci(filepath):
    if filepath[-3:] == 'xml':
        return parse_msci_file(filepath)
    elif filepath[-3:] == 'zip':
        return parse_msci_file(open_file_from_zip(filepath))
    else:
        print('Could not load:', filepath)


def parse_msci_file(filepath):
    return pd.DataFrame([child1.attrib for child0 in ET.parse(filepath).getroot() for child1 in child0])


def open_file_from_zip(filepath):
    return zipfile.ZipFile(filepath, 'r').open(filepath.split('/')[-1].upper()[:-3] + 'xml')


if __name__ == "__main__":
    filepath_1 = r'C:\Users\mnguyen\LGSS\Investments Team - SandPits - SandPits\data\input\test\xml\20200203_20200228_core_ap_index_main_monthly.zip'.replace(
        "\\", "/")
    df = load_msci(filepath_1)
