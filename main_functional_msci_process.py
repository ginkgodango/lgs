import zipfile
import os
import pandas as pd
import xml.etree.ElementTree as ET
import time
import multiprocessing


def process_msci_zip(instruction):
    """
    Process the MSCI zip files.
    :param instruction: (filename, input_directory, output_directory)
    :return:
    """
    filename, input_dir, output_dir = instruction

    create_folder_from_zip(filename, input_dir, output_dir)

    list(map(lambda x: io_msci(filename, input_dir, output_dir, x), create_xml_list_from_zip(filename, input_dir)))

    print('{timestamp} Processed: {output_directory}'.format(timestamp=time.ctime(), output_directory=folder_path(filename, output_dir)))


def folder_path(filename, directory):

    return directory + filename[:-4] + '/'


def create_folder_from_zip(filename, input_dir, output_dir):

    return (
        os.makedirs(folder_path(filename, output_dir)) if len(create_xml_list_from_zip(filename, input_dir)) > 0 else
        None
    )


def check_new_zip(s, output_dir):

    return s[:-4] not in os.listdir(output_dir)


def create_xml_list_from_zip(filename, input_dir):

    return sorted(list(filter(lambda x: x.endswith('.xml'), zipfile.ZipFile(input_dir + filename, 'r').namelist())))


def open_xml_from_zip(filename, input_dir, xml_file):

    return zipfile.ZipFile(input_dir + filename, 'r').open(xml_file)


def parse_xml_to_dataframe(file):
    """
    Parses the .xml file to dataframe file.
    :param filepath:
    :return:
    """

    return pd.DataFrame([child1.attrib for child0 in ET.parse(file).getroot() for child1 in child0])


def io_msci(zip_filename, input_dir, output_dir, xml_filename):

    xml_file = open_xml_from_zip(zip_filename, input_dir, xml_filename)

    output_path = folder_path(zip_filename, output_dir) + xml_filename[:-4] + '.csv'

    return parse_xml_to_dataframe(xml_file).to_csv(output_path, index=False)


if __name__ == '__main__':

    input_directory = 'C:/Users/Mnguyen/LGSS/Investments Team - SandPits - SandPits/data/input/vendors/msci/download/'

    output_directory = 'C:/Users/Mnguyen/Data/msci/csv/'

    files_zip = list(filter(lambda x: x.endswith('.zip'), sorted(os.listdir(input_directory))))

    files_zip_new = list(filter(lambda x: check_new_zip(x, output_directory), files_zip))

    process_instructions = list(map(lambda x: (x, input_directory, output_directory), files_zip_new))

    generate = multiprocessing.Pool(processes=10).imap(process_msci_zip, process_instructions)
