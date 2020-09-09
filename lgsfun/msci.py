import xml.etree.ElementTree as ET
import pandas as pd


def load_msci(filepath):
    return pd.DataFrame([child1.attrib for child0 in ET.parse(filepath).getroot() for child1 in child0])
