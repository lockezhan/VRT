# ##################################################################################################
#  The MIT License (MIT)
#  Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
# 
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software
#  and associated documentation files (the "Software"), to deal in the Software without restriction,
#  including without limitation the rights to use, copy, modify, merge, publish, distribute,
#  sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included in all copies or
#  substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ##################################################################################################

import xml.etree.ElementTree as ET
from xml.dom import minidom
import argparse

def read_input_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def add_instance(parent, instance, module, total_luts, logic_luts, lutrams, srls, ffs, ramb36, ramb18, uram, dsp_blocks):
    instance_element = ET.SubElement(parent, "Instance")
    ET.SubElement(instance_element, "Name").text = instance.strip()
    ET.SubElement(instance_element, "Module").text = module.strip()
    ET.SubElement(instance_element, "TotalLUTs").text = total_luts.strip()
    ET.SubElement(instance_element, "LogicLUTs").text = logic_luts.strip()
    ET.SubElement(instance_element, "LUTRAMs").text = lutrams.strip()
    ET.SubElement(instance_element, "SRLs").text = srls.strip()
    ET.SubElement(instance_element, "FFs").text = ffs.strip()
    ET.SubElement(instance_element, "RAMB36").text = ramb36.strip()
    ET.SubElement(instance_element, "RAMB18").text = ramb18.strip()
    ET.SubElement(instance_element, "URAM").text = uram.strip()
    ET.SubElement(instance_element, "DSPBlocks").text = dsp_blocks.strip()
    return instance_element

def main(resource_file):
    root = ET.Element("UtilizationReport")

    input_text = read_input_file(resource_file)

    table_start = input_text.find("+---------------------------+")
    table_end = input_text.rfind("+---------------------------+")
    table_text = input_text[table_start:table_end + len("+---------------------------+")]

    lines = table_text.splitlines()

    current_parents = {0: root}

    for line in lines[1:]:
        parts = line.split('|')
        print(parts)
        
        if len(parts) < 13:
            continue
        
        leading_whitespace = parts[1][:len(parts[1]) - len(parts[1].lstrip())]
        fields = [part.strip() for part in parts[1:]]

        if len(fields) < 12:
            continue

        instance = fields[0]
        module = fields[1]
        total_luts = fields[2]
        logic_luts = fields[3]
        lutrams = fields[4]
        srls = fields[5]
        ffs = fields[6]
        ramb36 = fields[7]
        ramb18 = fields[8]
        uram = fields[9]
        dsp_blocks = fields[10]

        level = len(leading_whitespace)
        print(f"Instance: {instance}, Level: {level}, Leading whitespace: '{leading_whitespace}'")

        if level not in current_parents:
            valid_levels = sorted(current_parents.keys())
            for valid_level in reversed(valid_levels):
                if valid_level < level:
                    level = valid_level + 1
                    break
            else:
                level = 0

        parent_level = max(level - 1, 0)
        current_parents[level] = add_instance(current_parents[parent_level], instance, module, total_luts, logic_luts, lutrams, srls, ffs, ramb36, ramb18, uram, dsp_blocks)
        current_parents[level + 1] = current_parents[level]

    tree = ET.ElementTree(root)

    xml_str = ET.tostring(root, encoding='utf-8')

    parsed_xml = minidom.parseString(xml_str)
    pretty_xml_str = parsed_xml.toprettyxml(indent="  ")

    with open("build/utilization_report.xml", "w", encoding="utf-8") as f:
        f.write(pretty_xml_str)

    print("XML file created successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate XML utilization report.')
    parser.add_argument('--resource_file', type=str, required=True, help='Path to the resource file')
    args = parser.parse_args()
    main(args.resource_file)