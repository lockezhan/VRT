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

#include "commands/resource_command.hpp"

ResourceCommand::ResourceCommand(const std::string& device) : device(device) {
    std::string amiHome = std::string(getenv("AMI_HOME"));
    if (amiHome.empty()) {
        std::cerr << "AMI_HOME environment variable is not set." << std::endl;
        exit(1);
    }
    char filePath[1024];
    sprintf(filePath, RESOURCE_UTILIZATION_FILE, amiHome.c_str(), device.c_str());
    parseXML(std::string(filePath));
    printResources();
}

void ResourceCommand::parseXML(const std::string& filename) {
    xmlDoc* doc = xmlReadFile(filename.c_str(), NULL, 0);
    if (doc == NULL) {
        std::cerr << "Failed to parse XML file: " << filename << std::endl;
        exit(1);
    }

    xmlNode* rootElement = xmlDocGetRootElement(doc);
    if (rootElement) {
        parseInstance(rootElement, rootInstance);
    }

    xmlFreeDoc(doc);
    xmlCleanupParser();
}

void ResourceCommand::parseInstance(xmlNode* node, Instance& instance) {
    std::regex re("(\\d+)\\((\\d+\\.\\d+)%\\)");
    std::smatch match;
    for (xmlNode* curNode = node; curNode; curNode = curNode->next) {
        if (curNode->type == XML_ELEMENT_NODE) {
            if (xmlStrcmp(curNode->name, BAD_CAST "UtilizationReport") == 0) {
                parseInstance(curNode->children, instance);
                continue;
            }
            if (xmlStrcmp(curNode->name, BAD_CAST "Instance") == 0) {
                Instance childInstance;
                for (xmlNode* child = curNode->children; child; child = child->next) {
                    if (child->type == XML_ELEMENT_NODE) {
                        std::string content = (const char*)xmlNodeGetContent(child);
                        if (xmlStrcmp(child->name, BAD_CAST "Name") == 0) {
                            childInstance.name = content;
                        } else if (xmlStrcmp(child->name, BAD_CAST "Module") == 0) {
                            childInstance.module = content;
                        } else if (xmlStrcmp(child->name, BAD_CAST "TotalLUTs") == 0) {
                            if (std::regex_search(content, match, re) && match.size() == 3) {
                                childInstance.totalLUTs.first = std::stoi(match.str(1));
                                childInstance.totalLUTs.second = std::stof(match.str(2));
                            } else {
                                childInstance.totalLUTs.first = 0;
                                childInstance.totalLUTs.second = 0.0;
                            }
                        } else if (xmlStrcmp(child->name, BAD_CAST "LogicLUTs") == 0) {
                            if (std::regex_search(content, match, re) && match.size() == 3) {
                                childInstance.logicLUTs.first = std::stoi(match.str(1));
                                childInstance.logicLUTs.second = std::stof(match.str(2));
                            } else {
                                childInstance.logicLUTs.first = 0;
                                childInstance.logicLUTs.second = 0.0;
                            }
                        } else if (xmlStrcmp(child->name, BAD_CAST "LUTRAMs") == 0) {
                            if (std::regex_search(content, match, re) && match.size() == 3) {
                                childInstance.lutRAMs.first = std::stoi(match.str(1));
                                childInstance.lutRAMs.second = std::stof(match.str(2));
                            } else {
                                childInstance.lutRAMs.first = 0;
                                childInstance.lutRAMs.second = 0.0;
                            }
                        } else if (xmlStrcmp(child->name, BAD_CAST "SRLs") == 0) {
                            if (std::regex_search(content, match, re) && match.size() == 3) {
                                childInstance.srls.first = std::stoi(match.str(1));
                                childInstance.srls.second = std::stof(match.str(2));
                            } else {
                                childInstance.srls.first = 0;
                                childInstance.srls.second = 0.0;
                            }
                        } else if (xmlStrcmp(child->name, BAD_CAST "FFs") == 0) {
                            if (std::regex_search(content, match, re) && match.size() == 3) {
                                childInstance.ffs.first = std::stoi(match.str(1));
                                childInstance.ffs.second = std::stof(match.str(2));
                            } else {
                                childInstance.ffs.first = 0;
                                childInstance.ffs.second = 0.0;
                            }
                        } else if (xmlStrcmp(child->name, BAD_CAST "RAMB36") == 0) {
                            if (std::regex_search(content, match, re) && match.size() == 3) {
                                childInstance.ramb36.first = std::stoi(match.str(1));
                                childInstance.ramb36.second = std::stof(match.str(2));
                            } else {
                                childInstance.ramb36.first = 0;
                                childInstance.ramb36.second = 0.0;
                            }
                        } else if (xmlStrcmp(child->name, BAD_CAST "RAMB18") == 0) {
                            if (std::regex_search(content, match, re) && match.size() == 3) {
                                childInstance.ramb18.first = std::stoi(match.str(1));
                                childInstance.ramb18.second = std::stof(match.str(2));
                            } else {
                                childInstance.ramb18.first = 0;
                                childInstance.ramb18.second = 0.0;
                            }
                        } else if (xmlStrcmp(child->name, BAD_CAST "URAM") == 0) {
                            if (std::regex_search(content, match, re) && match.size() == 3) {
                                childInstance.uram.first = std::stoi(match.str(1));
                                childInstance.uram.second = std::stof(match.str(2));
                            } else {
                                childInstance.uram.first = 0;
                                childInstance.uram.second = 0.0;
                            }
                        } else if (xmlStrcmp(child->name, BAD_CAST "DSPBlocks") == 0) {
                            if (std::regex_search(content, match, re) && match.size() == 3) {
                                childInstance.dspBlocks.first = std::stoi(match.str(1));
                                childInstance.dspBlocks.second = std::stof(match.str(2));
                            } else {
                                childInstance.dspBlocks.first = 0;
                                childInstance.dspBlocks.second = 0.0;
                            }
                        }
                    }
                }
                parseInstance(curNode->children, childInstance);
                instance.children.push_back(childInstance);
            }
        }
    }
}

void ResourceCommand::printResources() {
    bool headerPrinted = false;
    std::cout << "Total usage" << std::endl;
    printSpecificInstance(rootInstance, "top_wrapper", headerPrinted);
    std::cout << "Base logic usage. AVED + VRT + User kernels" << std::endl;
    printSpecificInstance(rootInstance, "base_logic", headerPrinted);
    const std::vector<std::string> excludeList = {"axi_smbus_rpu",     "gcq_m2r", "hw_discovery",
                                                  "pcie_slr0_mgmt_sc", "rpu_sc",  "uuid_rom",
                                                  "clk_wiz",           "sys_rst", "noc_xbar"};
    std::cout << "User kernels usage" << std::endl;
    printChildrenOfBaseLogic(rootInstance, excludeList, headerPrinted);
}

void ResourceCommand::printInstance(const Instance& instance, int level,
                                    bool& headerPrinted) const {
    if (!headerPrinted) {
        // Print the table header
        printf(
            "+--------------------+--------------------+--------------------+--------------------+-"
            "-------------------+--------------------+--------------------+\n");
        printf(
            "| Name               | TotalLUTs          | LogicLUTs          | FFs                | "
            "BlockRAM Tiles     | URAM               | DSPBlocks          |\n");
        printf(
            "+--------------------+--------------------+--------------------+--------------------+-"
            "-------------------+--------------------+--------------------+\n");
        headerPrinted = true;
    }

    // Helper function to format the value and percentage
    auto formatValue = [](int value, float percentage) {
        std::ostringstream oss;
        oss << value << "(" << std::fixed << std::setprecision(2) << percentage << "%)";
        return oss.str();
    };

    // Print the instance details
    printf("| %-18s | %-18s | %-18s | %-18s | %-18s | %-18s | %-18s |\n", instance.name.c_str(),
           formatValue(instance.totalLUTs.first, instance.totalLUTs.second).c_str(),
           formatValue(instance.logicLUTs.first, instance.logicLUTs.second).c_str(),
           formatValue(instance.ffs.first, instance.ffs.second).c_str(),
           formatValue(instance.ramb36.first + instance.ramb18.first / 2,
                       instance.ramb36.second + instance.ramb18.second / 2)
               .c_str(),
           formatValue(instance.uram.first, instance.uram.second).c_str(),
           formatValue(instance.dspBlocks.first, instance.dspBlocks.second).c_str());

    printf(
        "+--------------------+--------------------+--------------------+--------------------+-----"
        "---------------+--------------------+--------------------+\n");
}
void ResourceCommand::printSpecificInstance(const Instance& instance, const std::string& targetName,
                                            bool headerPrinted) const {
    if (instance.name == targetName) {
        printInstance(instance, 0, headerPrinted);
        return;
    }
    for (const auto& child : instance.children) {
        printSpecificInstance(child, targetName, headerPrinted);
    }
}

void ResourceCommand::printChildrenOfBaseLogic(const Instance& instance,
                                               const std::vector<std::string>& excludeList,
                                               bool& headerPrinted) const {
    if (instance.name == "base_logic") {
        for (const auto& child : instance.children) {
            if (std::find(excludeList.begin(), excludeList.end(), child.name) ==
                excludeList.end()) {
                printInstance(child, 1, headerPrinted);
            }
        }
        return;
    }
    for (const auto& child : instance.children) {
        printChildrenOfBaseLogic(child, excludeList, headerPrinted);
    }
}