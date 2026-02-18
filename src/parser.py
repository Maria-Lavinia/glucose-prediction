import xml.etree.ElementTree as ET
import pandas as pd

def parse_xml_to_dataframe(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    glucose_section = root.find("glucose_level")

    data = []
    for entry in glucose_section:
        timestamp = entry.attrib["ts"]
        value = float(entry.attrib["value"])
        data.append([timestamp, value])

    df = pd.DataFrame(data, columns=["timestamp", "glucose"])

    df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True)
    df = df.sort_values("timestamp")
    df = df.set_index("timestamp")

    return df
