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

#ifndef REGISTER_HPP
#define REGISTER_HPP

#include <cstdint>
#include <iostream>
#include <string>

#include "logger.hpp"

/**
 * @brief Class representing a hardware register definition.
 *
 * This class stores information about a register within a memory-mapped component,
 * including its name, offset within the component, bit width, access permissions,
 * and a description of its functionality.
 */
class Register {
    std::string registerName;  ///< Name of the register
    uint32_t offset;           ///< Offset from the base address in bytes
    uint32_t width;            ///< Width of the register in bits
    std::string rw;            ///< Read/write permissions (e.g., "RW", "RO", "WO")
    std::string description;   ///< Description of the register's functionality

   public:
    /**
     * @brief Constructor for Register with initialization parameters.
     * @param registerName Name of the register.
     * @param offset Offset from the component's base address in bytes.
     * @param width Width of the register in bits.
     * @param rw Read/write permissions (e.g., "RW", "RO", "WO").
     * @param description Description of the register's functionality.
     */
    Register(std::string registerName, uint32_t offset, uint32_t width, std::string rw,
             std::string description);

    /**
     * @brief Default constructor for Register.
     */
    Register() = default;

    /**
     * @brief Gets the name of the register.
     * @return String containing the register name.
     */
    std::string getRegisterName();

    /**
     * @brief Gets the offset of the register.
     * @return Offset from the component's base address in bytes.
     */
    uint32_t getOffset();

    /**
     * @brief Gets the width of the register.
     * @return Width of the register in bits.
     */
    uint32_t getWidth();

    /**
     * @brief Gets the read/write permissions of the register.
     * @return String containing the access permissions (e.g., "RW", "RO", "WO").
     */
    std::string getRW();

    /**
     * @brief Gets the description of the register.
     * @return String containing the register's functional description.
     */
    std::string getDescription();

    /**
     * @brief Sets the name of the register.
     * @param registerName Name of the register to set.
     */
    void setRegisterName(std::string registerName);

    /**
     * @brief Sets the offset of the register.
     * @param offset Offset from the component's base address in bytes.
     */
    void setOffset(uint32_t offset);

    /**
     * @brief Sets the width of the register.
     * @param width Width of the register in bits.
     */
    void setWidth(uint32_t width);

    /**
     * @brief Sets the read/write permissions of the register.
     * @param rw Read/write permissions (e.g., "RW", "RO", "WO").
     */
    void setRW(std::string rw);

    /**
     * @brief Sets the description of the register.
     * @param description Description of the register's functionality.
     */
    void setDescription(std::string description);

    /**
     * @brief Prints the register information to the console.
     *
     * This method outputs the register's details including name, offset,
     * width, permissions and description for debugging or documentation purposes.
     */
    void print();
};

#endif  // REGISTER_HPP