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

    return list(map(lambda x: (x, read_msci(directory + '/' + x)), os.listdir(directory)))


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

    return list(filter(lambda x: x[-3:] == 'xml', zipfile.ZipFile(filepath, 'r').namelist()))


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

    return list(map(lambda x: zipfile.ZipFile(filepath, 'r').open(x), create_xml_list_from_zip(filepath)))


def parse_all_xml_to_dataframe(filepath):
    """
    Unused atm
    :param filepath: filepath to the .zip file
    :return: list containing dataframes parsed from all .xml files from a .zip file
    """

    return list(map(lambda x: parse_xml_to_dataframe(x), open_all_xml_from_zip(filepath)))


def file_matches(s, l):
    """

    :param s: filename
    :param l: list of filepaths
    :return: list of matching filepaths
    """
    return list(filter(lambda x: x.endswith(s), l))


def backup(directory):
    df_msci_dictionary = pd.read_excel(
        pd.ExcelFile(directory),
        sheet_name='MSCI XML Files',
        skiprows=[0, 1]
    )

    xml_package_names_0 = set(df_msci_dictionary['XML Package Name (1)'])
    xml_package_names_1 = set(filter(lambda x: x[9:17] == 'yyyymmdd', xml_package_names_0))
    xml_package_names_2 = set(filter(lambda x: x[0:8] == 'yyyymmdd', xml_package_names_0))
    xml_package_names_3 = set(filter(lambda x: x[0:8] != 'yyyymmdd', xml_package_names_0))
    xml_package_names_4 = xml_package_names_2 - xml_package_names_1
    xml_package_names_5 = set(list(map(lambda x: x[17:], xml_package_names_1)) + list(map(lambda x: x[8:], xml_package_names_4)))

    zipfiles = list(filter(lambda x: x.endswith(tuple(xml_package_names_5)), os.listdir(directory)))

    return zipfiles

def unzip(filename, input_dir, output_dir):

    return zipfile.ZipFile(input_dir + filename, 'r').extractall(folder_path(filename, output_dir))