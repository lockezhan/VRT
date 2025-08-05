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

import re
import xml.etree.ElementTree as ET
import argparse

def extract_clkout1_primitive(filename):
    with open(filename, 'r') as file:
        for line in file:
            if 'clkout1_primitive' in line:
                match = re.search(r'clkout1_primitive\s+([\d.]+)', line)
                if match:
                    return float(match.group(1))
    return None

def update_clock_frequency(xml_filename, clkout1_primitive_value):
    tree = ET.parse(xml_filename)
    root = tree.getroot()

    clock_frequency_element = root.find('ClockFrequency')
    if clock_frequency_element is not None:
        current_frequency_hz = int(clock_frequency_element.text)

        current_period_ns = 1e9 / current_frequency_hz

        new_period_ns = current_period_ns - clkout1_primitive_value

        new_frequency_hz = int(1e9 / new_period_ns)

        clock_frequency_element.text = str(new_frequency_hz)

        tree.write(xml_filename)
        return new_frequency_hz
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Update ClockFrequency in system map XML based on timing report.')
    parser.add_argument('--system_map', required=True, help='Path to the system map XML file')
    parser.add_argument('--timing', required=True, help='Path to the timing report file')

    args = parser.parse_args()

    clkout1_primitive_value = extract_clkout1_primitive(args.timing)
    if clkout1_primitive_value is not None:
        print(f'clkout1_primitive value: {clkout1_primitive_value}')

        new_frequency_hz = update_clock_frequency(args.system_map, clkout1_primitive_value)
        if new_frequency_hz is not None:
            print(f'Updated ClockFrequency value: {new_frequency_hz} Hz')
        else:
            print('ClockFrequency element not found in the XML file')
    else:
        print('clkout1_primitive value not found in the timing report')