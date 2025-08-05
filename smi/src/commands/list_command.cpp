/**
 * The MIT License (MIT)
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software
 * and associated documentation files (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge, publish, distribute,
 * sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or
 * substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
 * NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "commands/list_command.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

ListCommand::ListCommand(uint16_t vendorId, uint16_t deviceId)
    : vendorId(vendorId), deviceId(deviceId) {}

void ListCommand::execute() const { listDevices(); }

void ListCommand::listDevices() const {
    std::cout << "--------------------------------------------------------------------\n";
    std::cout << "Listing V80 devices "
              << "\n";
    std::cout << "--------------------------------------------------------------------\n";

    for (const auto& entry : std::filesystem::directory_iterator("/sys/bus/pci/devices")) {
        std::string path = entry.path();
        std::ifstream vendorFile(path + "/vendor");
        std::ifstream deviceFile(path + "/device");

        if (vendorFile.is_open() && deviceFile.is_open()) {
            std::string vendorIdStr, deviceIdStr;
            std::getline(vendorFile, vendorIdStr);
            std::getline(deviceFile, deviceIdStr);

            uint16_t vendorId = std::stoi(vendorIdStr, nullptr, 16);
            uint16_t deviceId = std::stoi(deviceIdStr, nullptr, 16);

            if (vendorId == this->vendorId && deviceId == this->deviceId) {
                std::cout << "V80 device found with BDF: " << entry.path().filename().string()
                          << "\n";
                std::cout
                    << "--------------------------------------------------------------------\n";
            }
        }
    }
}