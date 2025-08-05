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

#ifndef RESET_COMMAND_HPP
#define RESET_COMMAND_HPP

#include <ami.h>
#include <ami_mem_access.h>
#include <ami_program.h>

#include <exception>
#include <iostream>
#include <string>

#include "pcie_hotplug.hpp"

/**
 * @brief Class for handling device reset commands.
 *
 * The ResetCommand class provides functionality to reset a specified FPGA device
 * by managing the device's PCIe connection and hardware reset sequence.
 */
class ResetCommand {
   public:
    /**
     * @brief Constructor for ResetCommand.
     *
     * @param device The BDF (Bus:Device.Function) identifier of the device to reset.
     */
    ResetCommand(const std::string& device);

    /**
     * @brief Executes the reset command.
     *
     * This method performs the device reset sequence, which includes
     * properly shutting down the PCIe connection, performing the hardware
     * reset, and re-establishing the connection if necessary.
     */
    void execute();

   private:
    std::string device;  ///< The BDF identifier of the device to reset.
    ami_device* dev;     ///< Pointer to the AMI device object.
};

#endif  // RESET_COMMAND_HPP