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

#ifndef LIST_COMMAND_HPP
#define LIST_COMMAND_HPP

#include <string>

/**
 * @brief Class for listing available V80 devices.
 *
 * The ListCommand class provides functionality to list available devices
 * that match the specified vendor and device IDs.
 */
class ListCommand {
   public:
    /**
     * @brief Constructor for ListCommand.
     *
     * @param vendorId The vendor ID to filter devices by.
     * @param deviceId The device ID to filter devices by.
     */
    ListCommand(uint16_t vendorId, uint16_t deviceId);

    /**
     * @brief Executes the list command.
     *
     * This method lists all available devices that match the specified
     * vendor and device IDs.
     */
    void execute() const;

   private:
    uint16_t vendorId;  ///< The vendor ID to filter devices by.
    uint16_t deviceId;  ///< The device ID to filter devices by.

    /**
     * @brief Lists all matching devices.
     *
     * This method performs the actual listing of devices that match the
     * specified vendor and device IDs.
     */
    void listDevices() const;
};

#endif  // LIST_COMMAND_HPP