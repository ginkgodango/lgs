from multiprocessing import Pool
import xml.etree.ElementTree as ET
import pandas as pd
import zipfile
import os
import time
import datetime
import sys


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


def get_file_category(filename):
    """
    Extracts the file category from filename. E.g. 'yyyymmdd_yyyymmdd_category.zip' -> 'category.zip'
    :param filename: name of zip file
    :return: file category string
    """

    return (
        None if len(filename) <= 8 else
        filename[18:] if filename[8] == '_' and filename[18:] not in ('rif.zip', 'zip') else
        filename[8:] if filename[8:] not in ('rif.zip', 'zip') else
        None
    )


def read_combine_msci_files(filepaths):
    """

    :param filepaths:
    :return:
    """

    return pd.concat(list(map(read_msci, filepaths))).reset_index(drop=True)


def io_msci(category_filepaths_directory_tuple):
    """

    :param category_filepaths_directory_tuple:
    :return:
    """
    category, filepaths, directory = category_filepaths_directory_tuple

    try:

        print(datetime.datetime.now(), 'Processing', len(filepaths), 'files:', category)
        return read_combine_msci_files(filepaths).to_csv(directory + category[:-3] + 'csv', index=False)

    except ValueError:

        pass


if __name__ == "__main__":

    processors = 4

    input_directory = 'C:/Users/mnguyen/LGSS/Investments Team - SandPits - SandPits/data/input/test/mix/'

    output_directory = 'C:/Users/mnguyen/LGSS/Investments Team - SandPits - SandPits/data/output/vendors/msci/test/mix/'

    file_paths = set(map(lambda x: input_directory + x, os.listdir(input_directory)))

    file_categories = set(map(lambda x: str(get_file_category(x)), os.listdir(input_directory)))

    grouped_tuples = list(map(lambda x: (x, file_matches(x, file_paths), output_directory), file_categories))

    write = Pool(processes=processors).imap(io_msci, grouped_tuples)
