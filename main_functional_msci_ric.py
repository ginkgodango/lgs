import os
import pandas as pd


ric_folders_endswith = (
    'core_dm_daily_c_d_rif',
    'core_em_daily_d_rif',
    'scap_dm_daily_c_d_rif',
    'scap_em_daily_d_rif'
)

ric_files_endswith = (
    'CORE_DM_ALL_SECURITY_CODE_DAILY_D_RIF',
    'CORE_EM_ALL_SECURITY_CODE_DAILY_D_RIF',
    'SCAP_DM_ALL_SECURITY_CODE_DAILY_D_RIF',
    'SCAP_EM_ALL_SECURITY_CODE_DAILY_D_RIF'
)

if __name__ == '__main__':

    input_directory = 'C:/Users/Mnguyen/Data/msci/csv/'
    folder_list = os.listdir(input_directory)
    ric_folder_list = sorted(list(filter(lambda x: x.endswith(ric_folders_endswith), folder_list)))
