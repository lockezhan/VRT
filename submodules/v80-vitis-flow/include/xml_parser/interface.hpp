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

#pragma once

#include <cstdint>
#include <iostream>
#include <string>

#include "logger.hpp"

/**
 * @brief Class representing a hardware interface in a kernel.
 *
 * This class stores information about hardware interfaces such as AXI4, AXI-Lite,
 * or AXI-Stream interfaces that connect kernels to each other or to the system.
 * It includes properties like interface type, bus type, mode, and width settings.
 */
class Interface {
    std::string interfaceName;  ///< Name of the interface
    std::string interfaceType;  ///< Type of interface (e.g., "s_axilite", "m_axi")
    std::string busType;        ///< Type of bus (e.g., "axi4", "axi4lite")
    std::string mode;           ///< Mode of the interface (e.g., "master", "slave")
    uint8_t dataWidth;          ///< Width of the data bus in bits
    uint8_t addrWidth;          ///< Width of the address bus in bits

   public:
    /**
     * @brief Default constructor for Interface.
     */
    Interface() = default;

    /**
     * @brief Gets the name of the interface.
     * @return String containing the interface name.
     */
    std::string getInterfaceName();

    /**
     * @brief Sets the name of the interface.
     * @param name String containing the interface name.
     */
    void setInterfaceName(const std::string& name);

    /**
     * @brief Gets the type of the interface.
     * @return String containing the interface type.
     */
    std::string getInterfaceType();

    /**
     * @brief Sets the type of the interface.
     * @param type String containing the interface type.
     */
    void setInterfaceType(const std::string& type);

    /**
     * @brief Gets the bus type of the interface.
     * @return String containing the bus type.
     */
    std::string getBusType();

    /**
     * @brief Sets the bus type of the interface.
     * @param type String containing the bus type.
     */
    void setBusType(const std::string& type);

    /**
     * @brief Gets the mode of the interface.
     * @return String containing the interface mode.
     */
    std::string getMode();

    /**
     * @brief Sets the mode of the interface.
     * @param mode String containing the interface mode.
     */
    void setMode(const std::string& mode);

    /**
     * @brief Gets the data width of the interface.
     * @return Data width in bits.
     */
    uint8_t getDataWidth();

    /**
     * @brief Sets the data width of the interface.
     * @param width Data width in bits.
     */
    void setDataWidth(uint8_t width);

    /**
     * @brief Gets the address width of the interface.
     * @return Address width in bits.
     */
    uint8_t getAddrWidth();

    /**
     * @brief Sets the address width of the interface.
     * @param width Address width in bits.
     */
    void setAddrWidth(uint8_t width);

    /**
     * @brief Prints interface information.
     *
     * This method outputs the interface details including name, type,
     * bus type, mode, and width settings for debugging or documentation.
     */
    void print();
};