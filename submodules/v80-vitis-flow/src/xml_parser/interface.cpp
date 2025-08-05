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

#include "interface.hpp"

std::string Interface::getInterfaceName() { return this->interfaceName; }
void Interface::setInterfaceName(const std::string& intf_name) { this->interfaceName = intf_name; }

std::string Interface::getInterfaceType() { return this->interfaceType; }

void Interface::setInterfaceType(const std::string& intf_type) { this->interfaceType = intf_type; }

std::string Interface::getBusType() { return this->busType; }

void Interface::setBusType(const std::string& bus_type) { this->busType = bus_type; }

std::string Interface::getMode() { return this->mode; }

void Interface::setMode(const std::string& mode) { this->mode = mode; }

uint8_t Interface::getDataWidth() { return this->dataWidth; }

void Interface::setDataWidth(uint8_t dw) { this->dataWidth = dw; }

uint8_t Interface::getAddrWidth() { return this->addrWidth; }

void Interface::setAddrWidth(uint8_t aw) { this->addrWidth = aw; }

void Interface::print() {
    utils::Logger::log(utils::LogLevel::INFO, __PRETTY_FUNCTION__, "Interface name: {}",
                       interfaceName);
    utils::Logger::log(utils::LogLevel::INFO, __PRETTY_FUNCTION__, "Interface type: {}",
                       interfaceType);
    utils::Logger::log(utils::LogLevel::INFO, __PRETTY_FUNCTION__, "Bus type: {}", busType);
    utils::Logger::log(utils::LogLevel::INFO, __PRETTY_FUNCTION__, "Mode: {}", mode);
    if (busType == "aximm") {
        utils::Logger::log(utils::LogLevel::INFO, __PRETTY_FUNCTION__, "Data width: {}",
                           static_cast<int>(dataWidth));
        utils::Logger::log(utils::LogLevel::INFO, __PRETTY_FUNCTION__, "Addr width: {}",
                           static_cast<int>(addrWidth));
    }
}