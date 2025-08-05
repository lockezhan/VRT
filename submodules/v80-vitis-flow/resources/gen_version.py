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

import json
import re
import argparse
from datetime import datetime

def extract_logic_uuid(log_file):
    with open(log_file, 'r') as file:
        for line in file:
            match = re.search(r'Logic-UUID is ([a-f0-9]+)', line)
            if match:
                return match.group(1)
    raise ValueError("Logic-UUID not found in log file")

def create_json(log_file, name):
    logic_uuid = extract_logic_uuid(log_file)
    release = datetime.now().strftime("%Y%m%d")
    application = name

    data = {
        "design": {
            "name": name,
            "release": release,
            "logic_uuid": logic_uuid,
            "application": application
        }
    }

    with open('version.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create JSON from vivado.log and input parameters.')
    parser.add_argument('--log_file', type=str, required=True, help='Path to the vivado.log file')
    parser.add_argument('--name', type=str, required=True, help='Design name')

    args = parser.parse_args()

    create_json(args.log_file, args.name)