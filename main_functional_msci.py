import xml.etree.ElementTree as ET
import pandas as pd


def load_msci(filepath):
    return pd.DataFrame([child1.attrib for child0 in ET.parse(filepath).getroot() for child1 in child0])


if __name__ == "__main__":
    filepath_1 = r'C:\Users\mnguyen\LGSS\Investments Team - SandPits - SandPits\data\input\test\xml\20200203_20200228_core_ap_sec_main_daily_rif\20200203_20200228_CORE_AP_SEC_MAIN_DAILY_RIF.xml'.replace(
        "\\", "/")
    df = load_msci(filepath_1)
    df.to_csv(r'C:\Users\mnguyen\LGSS\Investments Team - SandPits - SandPits\data\input\test\xml\20200203_20200228_core_ap_sec_main_daily_rif\20200203_20200228_CORE_AP_SEC_MAIN_DAILY_RIF.csv'.replace(
        "\\", "/"), index=False)

