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

#include "api/kernel.hpp"

#include "api/device.hpp"

namespace vrt {

Kernel::Kernel(ami_device* device, const std::string& name, uint64_t baseAddr, uint64_t range,
               const std::vector<Register>& registers) {
    this->dev = device;
    this->name = name;
    this->baseAddr = baseAddr;
    this->range = range;
    this->registers = registers;
}

Kernel::Kernel(Device& device, const std::string& kernelName)
    : Kernel(device.getKernel(kernelName)) {
    deviceBdf = device.getBdf();
    this->platform = device.getPlatform();
    this->server = device.getZmqServer();
}

void Kernel::write(uint32_t offset, uint32_t value) {
    if (platform == Platform::HARDWARE) {
        utils::Logger::log(utils::LogLevel::DEBUG, __PRETTY_FUNCTION__,
                           "Writing to device {} kernel: {} at offset: {x} value: {x}", deviceBdf,
                           name, offset, value);
        uint32_t* buf = (uint32_t*)calloc(1, sizeof(uint32_t));
        *buf = value;
        if (buf) {
            int ret = ami_mem_bar_write(dev, bar, baseAddr - BASE_BAR_ADDR + offset, buf[0]);
            if (ret != AMI_STATUS_OK) {
                throw std::runtime_error("Failed to write to device");
            }
        }
        free(buf);
    } else if (platform == Platform::SIMULATION) {
        server->sendScalar(baseAddr + offset, value);
    }
}

uint32_t Kernel::read(uint32_t offset) {
    if (platform == Platform::HARDWARE) {
        if (offset != 0)
            utils::Logger::log(utils::LogLevel::DEBUG, __PRETTY_FUNCTION__,
                               "Reading from device {} kernel: {} at offset: {x}", deviceBdf, name,
                               offset);
        uint32_t* buf = (uint32_t*)calloc(1, sizeof(uint32_t));
        if (buf) {
            int ret = ami_mem_bar_read(dev, bar, baseAddr - BASE_BAR_ADDR + offset, &buf[0]);
            if (ret != AMI_STATUS_OK) {
                throw std::runtime_error("Failed to read from device");
            }
        }
        uint32_t value = buf[0];
        free(buf);
        return value;
    } else if (platform == Platform::EMULATION) {
        currentRegisterIndex = 4;
        std::size_t argIdx = 0;
        while (currentRegisterIndex < registers.size()) {
            std::regex re(".*_\\d+$");
            if (std::regex_match(registers.at(currentRegisterIndex).getRegisterName(), re)) {
                currentRegisterIndex += 2;
            } else {
                if (registers.at(currentRegisterIndex).getOffset() == offset) {
                    return server->fetchScalar(name, "arg" + std::to_string(argIdx));
                }
                currentRegisterIndex++;
            }
            argIdx++;
        }
    } else if (platform == Platform::SIMULATION) {
        return server->fetchScalarSim(baseAddr + offset);
    }
    return 0;
}

void Kernel::setDevice(ami_device* device) { this->dev = device; }

void Kernel::wait() {
    if (platform == Platform::EMULATION) {
        return;
    }
    while (read(0x00) == 1 || read(0x00) == 0x81) {
    }
}

void Kernel::startKernel(bool autorestart) {
    if (autorestart) {
        write(0x00, 0x81);
    } else {
        write(0x00, 0x01);
    }
}

Kernel::~Kernel() {}

void Kernel::setPlatform(Platform platform) { this->platform = platform; }

void Kernel::writeBatch() {
    uint32_t noOfPhysicalRegisters =
        (registers.at(registers.size() - 1).getOffset() + sizeof(uint32_t)) / sizeof(uint32_t);
    uint32_t* buf = (uint32_t*)calloc(noOfPhysicalRegisters, sizeof(uint32_t));
    for (std::size_t i = 4; i < noOfPhysicalRegisters; i++) {
        buf[i] = registerMap[i * sizeof(uint32_t)];
        // buf[i] = registerMap[registers.at(i).getOffset()];
        utils::Logger::log(utils::LogLevel::DEBUG, __PRETTY_FUNCTION__,
                           "Kernel {}, reg at offset {x}, value: {x}", name, i * sizeof(uint32_t),
                           buf[i]);
    }
    ami_mem_bar_write_range(dev, bar, baseAddr - BASE_BAR_ADDR, noOfPhysicalRegisters, buf);
    free(buf);
}
std::string Kernel::getName() const { return name; }

}  // namespace vrt
