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

#ifndef RELOAD_COMMAND_HPP
#define RELOAD_COMMAND_HPP

#include <iostream>
#include <string>

#include "pcie_hotplug.hpp"

/**
 * @brief Class for handling device reload commands.
 *
 * The ReloadCommand class provides functionality to reload a specified FPGA device
 * by managing the PCIe connection through a hot-plug cycle without performing a complete
 * hardware reset. This is useful for refreshing the device state or recovering from
 * certain error conditions.
 */
class ReloadCommand {
   public:
    /**
     * @brief Constructor for ReloadCommand.
     *
     * @param device The BDF (Bus:Device.Function) identifier of the device to reload.
     */
    ReloadCommand(const std::string& device);

    /**
     * @brief Executes the reload command.
     *
     * This method performs the device reload sequence, which typically involves
     * removing the device from the PCIe bus and then rescanning to rediscover it,
     * effectively reestablishing the connection without a full hardware reset.
     */
    void execute();

   private:
    std::string device;  ///< The BDF identifier of the device to reload.
};

#endif  // RELOAD_COMMAND_HPP