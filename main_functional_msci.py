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

    return sorted(list(filter(lambda x: x[-3:] == 'xml', zipfile.ZipFile(filepath, 'r').namelist())))


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

    processors = 10

    mtd = '202009'

    input_directory_download = 'C:/Users/Mnguyen/LGSS/Investments Team - SandPits - SandPits/data/input/vendors/msci/download/'

    input_directory_history = 'C:/Users/mnguyen/LGSS/Investments Team - SandPits - SandPits/data/input/vendors/msci/download/history/'

    output_directory_mtd = 'C:/Users/mnguyen/LGSS/Investments Team - SandPits - SandPits/data/output/vendors/msci/mtd/'

    output_directory_rest = 'C:/Users/mnguyen/LGSS/Investments Team - SandPits - SandPits/data/output/vendors/msci/rest/'

    files_mtd_download = list(filter(lambda x: x.startswith(mtd), os.listdir(input_directory_download)))

    files_mtd_history = list(filter(lambda x: x.startswith(mtd), os.listdir(input_directory_history)))

    files_rest_download = sorted(list(set(os.listdir(input_directory_download)) - set(files_mtd_download)))

    files_rest_history = sorted(list(set(os.listdir(input_directory_history)) - set(files_mtd_history)))

    file_paths_mtd_download = sorted(list(map(lambda x: input_directory_download + x, files_mtd_download)))

    file_paths_rest_download = sorted(list(map(lambda x: input_directory_download + x, files_rest_download)))

    file_paths_mtd_history = sorted(list(map(lambda x: input_directory_history + x, files_mtd_history)))

    file_paths_rest_history = sorted(list(map(lambda x: input_directory_history + x, files_rest_history)))

    file_paths_mtd = file_paths_mtd_download + file_paths_mtd_history

    file_paths_rest = file_paths_rest_download + file_paths_rest_history

    file_categories_mtd = sorted(list(set(map(lambda x: str(get_file_category(x)), files_mtd_download + files_mtd_history))))

    file_categories_rest = sorted(list(set(map(lambda x: str(get_file_category(x)), files_rest_download + files_rest_history))))

    process_instructions_mtd = list(map(lambda x: (x, file_matches(x, file_paths_mtd), output_directory_mtd), file_categories_mtd))

    process_instructions_rest = list(map(lambda x: (x, file_matches(x, file_paths_rest), output_directory_rest), file_categories_rest))

    # write_mtd = Pool(processes=processors).imap(io_msci, process_instructions_mtd)

    # write_rest = Pool(processes=processors).imap(io_msci, process_instructions_rest)
