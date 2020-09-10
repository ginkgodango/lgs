from multiprocessing import Pool
import xml.etree.ElementTree as ET
import pandas as pd
import zipfile
import os
import time

def read_msci(filepath):
    """
    Reads in a the filepath to the .zip file or .xml file and returns a dataframe of the MSCI data.
    :param filepath: filepath to the .zip file or .xml file.
    :return: dataframe containing the MSCI data
    """

    return (
        print('No .xml file found in zip file: ', filepath) if filepath[-3:] == 'zip' and create_xml_list_from_zip(filepath) == [] else
        parse_xml_to_dataframe(open_xml_from_zip(filepath)) if filepath[-3:] == 'zip' else
        parse_xml_to_dataframe(filepath) if filepath[-3:] == 'xml' else
        print('Could not load: ', filepath)
    )


def read_msci_folder(directory):
    """
    Reads in all files in directory
    :param directory: directory containing the msci .zip files
    :return: [(filename0, df0), (filename1, df1), ..., (filenameN, dfN)]
    """

    return list(map(lambda file: (file, read_msci(directory + '/' + file)), os.listdir(directory)))


# Helper Functions
def parse_xml_to_dataframe(filepath):
    """
    Parses the .xml file to dataframe file.
    :param filepath:
    :return:
    """

    return pd.DataFrame([child1.attrib for child0 in ET.parse(filepath).getroot() for child1 in child0])


def create_xml_list_from_zip(filepath):
    """
    Opens the .zip file and extracts a list of .xml filenames
    :param filepath: filepath to the .zip file
    :return: list of .xml filenames
    """

    return list(filter(lambda filename: filename[-3:] == 'xml', zipfile.ZipFile(filepath, 'r').namelist()))


def open_xml_from_zip(filepath):
    """
    Extracts .xml file from .zip file
    :param filepath: filepath to the .zip file
    :return: .xml file from .zip file
    """

    return zipfile.ZipFile(filepath, 'r').open(create_xml_list_from_zip(filepath)[0])


def open_all_xml_from_zip(filepath):
    """
    Unused atm
    :param filepath:filepath to the .zip file
    :return: list containing all .xml files from a .zip file
    """

    return list(map(lambda xml_name: zipfile.ZipFile(filepath, 'r').open(xml_name), create_xml_list_from_zip(filepath)))


def parse_all_xml_to_dataframe(filepath):
    """
    Unused atm
    :param filepath: filepath to the .zip file
    :return: list containing dataframes parsed from all .xml files from a .zip file
    """

    return list(map(lambda xml: parse_xml_to_dataframe(xml), open_all_xml_from_zip(filepath)))






if __name__ == "__main__":
    # filepath_1 = r'C:\Users\mnguyen\LGSS\Investments Team - SandPits - SandPits\data\input\test\xml\20200203_20200228_core_ap_index_main_monthly.zip'.replace("\\", "/")
    # xml_list = create_xml_list_from_zip(filepath_1)
    # df1 = read_msci(filepath_1)
    # df2 = parse_all_xml_to_dataframe(filepath_1)
    # directory_1 = r'C:\Users\mnguyen\LGSS\Investments Team - SandPits - SandPits\data\input\test\xml'.replace("\\", "/")
    # df3 = read_msci_folder(directory_1)
    # directory_2 = r'C:\Users\mnguyen\LGSS\Investments Team - SandPits - SandPits\data\input\test\legacy'.replace("\\", "/")
    # df4 = read_msci_folder(directory_2)

    directory_5 = r'C:\Users\mnguyen\LGSS\Investments Team - SandPits - SandPits\data\input\test\parallel'.replace("\\", "/")
    filepaths = [directory_5 + '/' + filename for filename in os.listdir(directory_5)]

    start_time = time.time()
    df5 = list(Pool(processes=1).map(read_msci, filepaths))
    end_time = time.time()
    print(end_time - start_time)

