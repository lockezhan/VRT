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

#ifndef RESOURCE_COMMAND_HPP
#define RESOURCE_COMMAND_HPP

#include <libxml/parser.h>
#include <libxml/tree.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <regex>
#include <string>
#include <utility>
#include <vector>

/**
 * @brief Format string for resource utilization file path.
 *
 * This macro defines the format string for the path to the resource utilization XML file.
 */
#define RESOURCE_UTILIZATION_FILE "%s/%s:00.0/report_utilization.xml"

/**
 * @brief Structure representing an instance in the resource hierarchy.
 *
 * This structure holds resource utilization data for a specific instance, including
 * its name, module type, resource counts, and child instances.
 */
struct Instance {
    std::string name;                 ///< Name of the instance.
    std::string module;               ///< Module type of the instance.
    std::pair<int, float> totalLUTs;  ///< Total number of LUTs used and percentage of total.
    std::pair<int, float> logicLUTs;  ///< Number of logic LUTs used and percentage of total.
    std::pair<int, float> lutRAMs;    ///< Number of LUT RAMs used and percentage of total.
    std::pair<int, float> srls;       ///< Number of SRLs used and percentage of total.
    std::pair<int, float> ffs;        ///< Number of flip-flops used and percentage of total.
    std::pair<int, float> ramb36;     ///< Number of RAMB36 blocks used and percentage of total.
    std::pair<int, float> ramb18;     ///< Number of RAMB18 blocks used and percentage of total.
    std::pair<int, float> uram;       ///< Number of URAMs used and percentage of total.
    std::pair<int, float> dspBlocks;  ///< Number of DSP blocks used and percentage of total.
    std::vector<Instance> children;   ///< Child instances.
};

/**
 * @brief Class for displaying resource utilization information.
 *
 * The ResourceCommand class provides functionality to display resource utilization
 * information for a specified device, showing detailed breakdowns of hardware resources
 * used by different components.
 */
class ResourceCommand {
   public:
    /**
     * @brief Constructor for ResourceCommand.
     *
     * @param device The BDF of the device to query for resource utilization.
     */
    ResourceCommand(const std::string& device);

    /**
     * @brief Executes the resource command.
     *
     * This method retrieves and displays resource utilization information for the specified device.
     */
    void execute() const;

   private:
    std::string device;     ///< The BDF of the device to query.
    Instance rootInstance;  ///< Root instance of the resource hierarchy.

    /**
     * @brief Parses the resource utilization XML file.
     *
     * @param filename Path to the XML file containing resource utilization data.
     *
     * This method parses the XML file and builds the resource hierarchy.
     */
    void parseXML(const std::string& filename);

    /**
     * @brief Parses a single instance node from the XML.
     *
     * @param a_node The XML node to parse.
     * @param instance The Instance structure to populate.
     *
     * This method parses a single XML node representing an instance and populates
     * the corresponding Instance structure.
     */
    void parseInstance(xmlNode* a_node, Instance& instance);

    /**
     * @brief Prints resource utilization information.
     *
     * This method displays formatted resource utilization information.
     */
    void printResources();

    /**
     * @brief Prints detailed information for a specific instance.
     *
     * @param instance The instance to print information for.
     * @param level The hierarchical level of the instance.
     * @param headerPrinted Reference to a flag indicating if the header has been printed.
     *
     * This method prints detailed resource utilization information for a specific instance.
     */
    void printInstance(const Instance& instance, int level, bool& headerPrinted) const;

    /**
     * @brief Prints information for an instance with a specific name.
     *
     * @param instance The root instance to search in.
     * @param targetName The name of the instance to print information for.
     * @param headerPrinted Flag indicating if the header has been printed.
     *
     * This method searches for and prints information for an instance with a specific name.
     */
    void printSpecificInstance(const Instance& instance, const std::string& targetName,
                               bool headerPrinted) const;

    /**
     * @brief Prints information for children of the base logic instance.
     *
     * @param instance The root instance to search in.
     * @param excludeList List of instance names to exclude from the output.
     * @param headerPrinted Reference to a flag indicating if the header has been printed.
     *
     * This method prints resource utilization information for children of the base logic
     * instance, excluding instances in the provided exclude list.
     */
    void printChildrenOfBaseLogic(const Instance& instance,
                                  const std::vector<std::string>& excludeList,
                                  bool& headerPrinted) const;
};

#endif  // RESOURCE_COMMAND_HPP