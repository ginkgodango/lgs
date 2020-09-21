"""
Note some functions are impure due to the nature of FTPlib and IO operations.
The present working directory for MSCI FTP is "../download"

Note: MSCI FTP has two folders to store files:
1. "../download"
2. "../download/history"
"""
from ftplib import FTP
from multiprocessing import Pool
import os
import shutil


def get_msci_folders(ftp):

    return set(ftp.nlst()) - set(filter(lambda x: x.endswith(('.zip', '.cis', '.h__')), ftp.nlst()))


def get_msci_files_download(ftp):

    ftp.cwd("../download/")

    return set(ftp.nlst())


def get_msci_files_history(ftp):

    ftp.cwd("../download/history")
    files = set(ftp.nlst())
    ftp.cwd("../")
    ftp.cwd("../download")

    return files


def get_new_msci_files_download(ftp, output_dir):
    """
    Finds the new files in the MSCI FTP "../download" directory that does not exist in LGS download folder.
    :param ftp:
    :param output_dir:
    :return:
    """

    ftp.cwd("../download")
    new_msci_files_download = set(ftp.nlst()) - set(os.listdir(output_dir))

    return new_msci_files_download


def move_files(move_instruction):
    """
    Unpacks a tuple of instructions and moves files from folder a to folder b
    :param move_instruction: (file, folder_a, folder_b)
    :return: Moves the file from folder a to folder b
    """
    file, folder_a, folder_b = move_instruction
    print("Moving:", file, "from", folder_a, "to", folder_b)

    return shutil.move(folder_a + file, folder_b + file)


def download_from_ftp(transfer_instruction):
    """

    :param transfer_instruction:
    :return:
    """
    website, username, password, file, ftp_directory, lgs_directory = transfer_instruction
    ftp = FTP(website)
    ftp.login(username, password)
    print("Transferring:", file, "from", ftp_directory, "to", lgs_directory)
    ftp.retrbinary("RETR " + ftp_directory + file, open(lgs_directory + str(file), 'wb').write)
    ftp.quit()


if __name__ == '__main__':
    processors = 10
    directory_download = 'C:/Users/Mnguyen/LGSS/Investments Team - SandPits - SandPits/data/input/vendors/msci/download/'
    directory_history = 'C:/Users/Mnguyen/LGSS/Investments Team - SandPits - SandPits/data/input/vendors/msci/download/history/'
    ftp = FTP("ftp.msci.com")
    ftp.login('wnvyzpng', 'hxcksyyx')
    print(ftp.getwelcome())
    print("MSCI FTP working directory: ", ftp.pwd())

    msci_download_folders = get_msci_folders(ftp)

    msci_files_download = get_msci_files_download(ftp) - msci_download_folders

    msci_files_history = get_msci_files_history(ftp)

    lgs_files_download = set(os.listdir(directory_download))

    lgs_files_history = set(os.listdir(directory_history))

    moved_files = (msci_files_history - lgs_files_history).intersection(lgs_files_download)

    new_files_download = sorted(list((msci_files_download - lgs_files_download)))

    new_files_history = sorted(list((msci_files_history - lgs_files_history) - moved_files))

    move_instructions = list(map(lambda x: (x, directory_download, directory_history), moved_files))

    mover = Pool(processes=processors).imap(move_files, move_instructions)

    transfer_instructions_download = list(map(lambda x: ("ftp.msci.com", 'wnvyzpng', 'hxcksyyx', x, "../download/", directory_download), new_files_download))

    transfer_instructions_history = list(map(lambda x: ("ftp.msci.com", 'wnvyzpng', 'hxcksyyx', x, "../download/history/", directory_history), new_files_history))

    transfer_download = Pool(processes=processors).imap(download_from_ftp, transfer_instructions_download)

    transfer_history = Pool(processes=processors).imap(download_from_ftp, transfer_instructions_history)
