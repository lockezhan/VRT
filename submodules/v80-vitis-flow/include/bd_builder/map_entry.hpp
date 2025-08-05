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

#ifndef MAP_ENTRY_HPP
#define MAP_ENTRY_HPP

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "register.hpp"

/**
 * @brief Class representing a memory map entry for a hardware component.
 *
 * This class stores information about a memory-mapped component in a hardware design,
 * including its base address, address range, and register definitions. It is used
 * to document the memory map of kernels and other components in the system.
 */
class MapEntry {
    std::string name;                 ///< Name of the memory-mapped component
    uint64_t baseAddress;             ///< Base address of the component in the memory map
    uint64_t range;                   ///< Size of the address range in bytes
    std::vector<Register> registers;  ///< List of registers within this component

   public:
    /**
     * @brief Default constructor for MapEntry.
     */
    MapEntry() = default;

    /**
     * @brief Constructor for MapEntry with initialization parameters.
     * @param name Name of the memory-mapped component.
     * @param base_addr Base address of the component in the system memory map.
     * @param range Size of the address range in bytes.
     */
    MapEntry(std::string name, uint64_t base_addr, uint64_t range);

    /**
     * @brief Gets the name of the memory-mapped component.
     * @return String containing the component name.
     */
    std::string getName();

    /**
     * @brief Gets the base address of the component.
     * @return Base address as a 64-bit unsigned integer.
     */
    uint64_t getBaseAddr();

    /**
     * @brief Gets the size of the address range.
     * @return Range size in bytes as a 64-bit unsigned integer.
     */
    uint64_t getRange();

    /**
     * @brief Gets the list of registers within this component.
     * @return Vector of Register objects defining the component's registers.
     */
    std::vector<Register> getRegisters();

    /**
     * @brief Sets the list of registers for this component.
     * @param registers Vector of Register objects to associate with this component.
     */
    void setRegisters(std::vector<Register> registers);
};

#endif  // MAP_ENTRY_HPP