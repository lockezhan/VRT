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

#ifndef SYSTEM_MAP_HPP
#define SYSTEM_MAP_HPP

#include <libxml/parser.h>
#include <libxml/tree.h>

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "arg_parser.hpp"
#include "map_entry.hpp"

/**
 * @brief Class representing the complete system memory map for a hardware design.
 *
 * This class manages memory map entries for all components in the system and provides
 * functionality to export the system map to files for documentation and runtime use.
 * It also tracks streaming connections and clock frequency settings.
 */
class SystemMap {
    std::vector<MapEntry> entries;  ///< List of memory map entries for all components
    std::vector<StreamingConnection> qdmaStreamConnections;  ///< List of QDMA streaming connections
    uint64_t targetClockFreq;                                ///< Target clock frequency in Hz
    std::string SYSTEM_MAP_OUTPUT = "system.map";  ///< Output file name for the system map
    Platform platform;  ///< Target platform (hardware, simulation, emulation)

   public:
    /**
     * @brief Constructor for SystemMap.
     * @param platform Target platform (hardware, simulation, emulation).
     */
    SystemMap(Platform platform);

    /**
     * @brief Adds a memory map entry to the system map.
     * @param entry MapEntry object to add to the system map.
     */
    void addEntry(MapEntry entry);

    /**
     * @brief Writes the system map to output files.
     *
     * This method exports the complete system memory map to the output file specified
     * in SYSTEM_MAP_OUTPUT for use by runtime software and documentation.
     */
    void printToFile();

    /**
     * @brief Converts an integer value to a hexadecimal string representation.
     * @param value Integer value to convert.
     * @return String containing the hexadecimal representation.
     */
    static std::string intToHex(uint64_t value);

    /**
     * @brief Sets the target clock frequency for the system.
     * @param freq Clock frequency in Hz.
     */
    void setClockFreq(uint64_t freq);

    /**
     * @brief Adds a streaming connection to the system map.
     * @param connection StreamingConnection object to add to the system.
     */
    void addStreamConnection(StreamingConnection connection);

    /**
     * @brief Gets the list of memory map entries in the system.
     * @return Vector of MapEntry objects in the system map.
     */
    std::vector<MapEntry> getEntries();

    /**
     * @brief Gets the list of QDMA streaming connections.
     * @return Vector of StreamingConnection objects in the system.
     */
    std::vector<StreamingConnection> getStreamConnections();

    /**
     * @brief Gets the target clock frequency.
     * @return Target clock frequency in Hz.
     */
    uint64_t getClockFreq();

    /**
     * @brief Checks if the design is segmented.
    /**
     * @brief Gets the target platform.
     * @return Target platform (hardware, simulation, emulation).
     */
    Platform getPlatform();
};

#endif  // SYSTEM_MAP_HPP