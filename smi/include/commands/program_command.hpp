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

#ifndef PROGRAM_COMMAND_HPP
#define PROGRAM_COMMAND_HPP

#include <ami.h>
#include <ami_mem_access.h>
#include <ami_program.h>
#include <unistd.h>

#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include "arg_parser.hpp"
#include "pcie_hotplug.hpp"
#include "utils/vrtbin.hpp"

/**
 * @brief Enumeration for image types.
 *
 * This enum represents the different types of images that can be programmed.
 */
enum class ImageType {
    PDI,     ///< PDI (Platform Device Image) type
    VRTBIN,  ///< VRTBIN (VRT Binary) type
};

/**
 * @brief Class for handling program commands.
 *
 * The ProgramCommand class provides functionality to program a device
 * with a specified image file.
 */
class ProgramCommand {
   public:
    /**
     * @brief Constructor for ProgramCommand.
     *
     * @param device The BDF of the device to program.
     * @param image_path Path to the image file.
     * @param partition The partition number to program.
     */
    ProgramCommand(const std::string& device, const std::string& image_path, uint8_t partition);

    /**
     * @brief Executes the program command.
     *
     * This method programs the specified device with the image file.
     */
    void execute();

    /**
     * @brief Boots the device.
     *
     * This method initiates the boot process for the device after programming.
     */
    void bootDevice();

   private:
    std::string device;     ///< The BDF of the device to program.
    std::string imagePath;  ///< Path to the image file.
    uint8_t partition;      ///< The partition number to program.
    ami_device* dev;        ///< Pointer to the AMI device object.
};

#endif  // PROGRAM_COMMAND_HPP