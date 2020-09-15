from multiprocessing import Pool
import xml.etree.ElementTree as ET
import pandas as pd
import zipfile
import os
import time
import datetime


def read_msci(filepath):
    """
    Reads in a the filepath to the .zip file or .xml file and returns a dataframe of the MSCI data.
    :param filepath: filepath to the .zip file or .xml file.
    :return: dataframe containing the MSCI data
    """

    return (
        None if filepath[-3:] == 'zip' and create_xml_list_from_zip(filepath) == [] else
        parse_xml_to_dataframe(open_xml_from_zip(filepath)) if filepath[-3:] == 'zip' else
        parse_xml_to_dataframe(filepath) if filepath[-3:] == 'xml' else
        print('Could not load: ', filepath)
    )


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


def file_matches(s, l):
    """

    :param s: filename
    :param l: list of filepaths
    :return: list of matching filepaths
    """

    return list(filter(lambda x: x.endswith(s), l))


if __name__ == "__main__":

    directory = 'C:/Users/mnguyen/LGSS/Investments Team - SandPits - SandPits/data/input/test/parallel'

    file_paths = set(map(lambda x: directory + '/' + x, os.listdir(directory)))

    files_1 = list(filter(lambda x: x[8] == '_', os.listdir(directory)))

    files_2 = list(filter(lambda x: x[8] != '_', os.listdir(directory)))

    file_categories = set(list(map(lambda x: x[18:], files_1)) + list(map(lambda x: x[8:], files_2)))

    file_paths_grouped = list(map(lambda x: (x, file_matches(x, file_paths)), file_categories))

    for files in file_paths_grouped:

        try:

            print(datetime.datetime.now(), 'Processing', len(files[1]), 'files:', files[1])

            df = pd.concat(list(Pool(processes=4).map(read_msci, files[1]))).reset_index(drop=True)

        except ValueError:

            continue
