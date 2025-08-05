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
#include <string>
#include <vector>

#include "logger.hpp"
#include "resource.hpp"

/**
 * @brief Class representing FPGA resource utilization and availability estimates.
 *
 * This class stores information about FPGA resource usage including both the
 * resources consumed by a design and the total available resources on the target
 * device. It supports tracking multiple resource types such as LUTs, FFs, BRAMs, etc.
 */
class AreaEstimates {
    std::vector<Resource> usedResources;       ///< Resources used by the design
    std::vector<Resource> availableResources;  ///< Total resources available on the target device

   public:
    /**
     * @brief Default constructor for AreaEstimates.
     */
    AreaEstimates() = default;

    /**
     * @brief Adds a used resource to the estimates.
     * @param resource Resource object representing a used FPGA resource.
     */
    void addResource(Resource resource);

    /**
     * @brief Adds an available resource to the estimates.
     * @param resource Resource object representing an available FPGA resource.
     */
    void addAvailableResource(Resource resource);

    /**
     * @brief Gets the list of used resources.
     * @return Vector of Resource objects representing used FPGA resources.
     */
    std::vector<Resource> getUsedResources();

    /**
     * @brief Gets the list of available resources.
     * @return Vector of Resource objects representing available FPGA resources.
     */
    std::vector<Resource> getAvailableResources();

    /**
     * @brief Prints the resource utilization information.
     *
     * This method outputs both used and available resources along with
     * utilization percentages for each resource type.
     */
    void print();
};