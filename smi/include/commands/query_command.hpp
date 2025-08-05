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

#ifndef QUERY_COMMAND_HPP
#define QUERY_COMMAND_HPP

#include <ami.h>
#include <ami_device.h>
#include <ami_mfg_info.h>
#include <fcntl.h>
#include <libxml/parser.h>
#include <libxml/tree.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "utils/vrtbin.hpp"

/**
 * @brief Path format for system map XML file.
 *
 * This macro defines the format string for the path to the system map XML file.
 */
#define SYSTEM_MAP_PATH "%s/%s:00.0/system_map.xml"

/**
 * @brief Path format for version JSON file.
 *
 * This macro defines the format string for the path to the version JSON file.
 */
#define VERSION_PATH "%s/%s:00.0/version.json"

/**
 * @brief Path format for QDMA queue device file.
 *
 * This macro defines the format string for the path to the QDMA queue device file.
 */
#define QDMA_QUEUE_PATH "/dev/qdma%s001-MM-0"  // /dev/qdma<bus><dev><func>-MM-<queue>

/**
 * @brief Path format for QDMA queue maximum value file.
 *
 * This macro defines the format string for the path to the QDMA queue maximum value file.
 */
#define QDMA_QMAX_PATH "/sys/bus/pci/devices/0000:%s:00.1/qdma/qmax"

/**
 * @brief Buffer size for reading files.
 *
 * This macro defines the size of the buffer used for reading files.
 */
#define BUFFER_SIZE 1024

/**
 * @brief Maximum length for metadata strings.
 *
 * This macro defines the maximum length for metadata strings.
 */
#define META_MAX_STR_LEN 256

/**
 * @brief Base for manufacturing timestamp conversion.
 *
 * This macro defines the base value for converting manufacturing timestamps.
 */
#define MFG_TIMESTAMP_BASE 16

/**
 * @brief Base year offset for manufacturing date.
 *
 * This macro defines the base year offset (1970) for manufacturing date calculations.
 */
#define MFG_DATE_TM_YEAR 70

/**
 * @brief Class for querying device information.
 *
 * The QueryCommand class provides functionality to query and display
 * information about a specified device, including its hardware details,
 * kernels, and QDMA queues.
 */
class QueryCommand {
   public:
    /**
     * @brief Constructor for QueryCommand.
     *
     * @param device The BDF of the device to query.
     */
    QueryCommand(const std::string& device);

    /**
     * @brief Executes the query command.
     *
     * This method queries the specified device and displays its information.
     */
    void execute();

   private:
    std::string device;  ///< The BDF of the device to query.

    /**
     * @brief Queries basic device information.
     *
     * This method retrieves and displays basic information about the device.
     */
    void queryDevice();

    /**
     * @brief Queries kernel information.
     *
     * @param bdf The BDF of the device to query.
     *
     * This method retrieves and displays information about the kernels on the device.
     */
    void queryKernels(const std::string& bdf);

    /**
     * @brief Queries QDMA queue information.
     *
     * @param bdf The BDF of the device to query.
     *
     * This method retrieves and displays information about the QDMA queues on the device.
     */
    void queryQueues(const std::string& bdf);

    /**
     * @brief Prints AMI-specific device details.
     *
     * This method retrieves and displays AMI-specific information about the device.
     */
    void printAmiDetails();

    /**
     * @brief Formats the manufacturing date.
     *
     * @param timestamp The manufacturing timestamp to format.
     *
     * This method formats and displays the manufacturing date based on the provided timestamp.
     */
    void formatManufacturingDate(long timestamp);
};

#endif  // QUERY_COMMAND_HPP