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
 * @brief Class representing a hardware resource in an FPGA design.
 *
 * This class stores information about a hardware resource such as LUTs,
 * FFs, BRAMs, or DSPs, including its name and numeric value (quantity).
 * It is used for tracking resource utilization in FPGA designs.
 */
class Resource {
    std::string name;  ///< Name of the resource type (e.g., "LUT", "FF", "BRAM", "DSP")
    uint32_t value;    ///< Quantity of the resource (count)

   public:
    /**
     * @brief Default constructor for Resource.
     */
    Resource() = default;

    /**
     * @brief Constructor for Resource with initialization parameters.
     * @param name String containing the resource type name.
     * @param value Quantity of the resource.
     */
    Resource(const std::string& name, uint32_t value) : name(name), value(value) {}

    /**
     * @brief Gets the name of the resource.
     * @return String containing the resource type name.
     */
    std::string getName();

    /**
     * @brief Sets the name of the resource.
     * @param name String containing the resource type name.
     */
    void setName(const std::string& name);

    /**
     * @brief Gets the quantity of the resource.
     * @return Unsigned integer representing the resource quantity.
     */
    uint32_t getValue();

    /**
     * @brief Sets the quantity of the resource.
     * @param value Unsigned integer representing the resource quantity.
     */
    void setValue(uint32_t value);

    /**
     * @brief Prints the resource information.
     *
     * This method outputs the resource type and quantity for
     * debugging or documentation purposes.
     */
    void print();
};