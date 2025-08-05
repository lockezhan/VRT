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

#include "xml_parser.hpp"

#include <cstring>

#include "defines.hpp"

std::string XmlParser::convertFromXmlCharPtr(const xmlChar* str) {
    if (str == nullptr) {
        return std::string();
    } else {
        return std::string(reinterpret_cast<const char*>(str));
    }
}

XmlParser::XmlParser(const std::string& fileName) {
    this->fileName = fileName;
    this->document = xmlReadFile(fileName.c_str(), nullptr, 0);
    if (this->document == nullptr) {
        throw std::runtime_error("Error: Unable to parse the XML file");
    }
    rootNode = xmlDocGetRootElement(document);
    workingNode = rootNode->children;
}

void XmlParser::parseXml() {
    for (auto current_node = workingNode; current_node; current_node = current_node->next) {
        if (current_node->type == XML_ELEMENT_NODE) {
            xmlChar* content = xmlNodeGetContent(current_node);
            std::string contentStr = convertFromXmlCharPtr(content);
            std::string nodeNameStr = convertFromXmlCharPtr(current_node->name);
            // std::cout << "Node Name: " << nodeNameStr << ", Content: " << contentStr <<
            // std::endl;
            if (nodeNameStr == XML_NODE_REGISTER) {
                xmlChar* offset = xmlGetProp(current_node, BAD_CAST "offset");
                xmlChar* name = xmlGetProp(current_node, BAD_CAST "name");
                xmlChar* access = xmlGetProp(current_node, BAD_CAST "access");
                xmlChar* description = xmlGetProp(current_node, BAD_CAST "description");
                xmlChar* range = xmlGetProp(current_node, BAD_CAST "range");

                std::string offsetStr = convertFromXmlCharPtr(offset);
                std::string nameStr = convertFromXmlCharPtr(name);
                std::string accessStr = convertFromXmlCharPtr(access);
                std::string descriptionStr = convertFromXmlCharPtr(description);
                std::string rangeStr = convertFromXmlCharPtr(range);
                Register r;
                r.setRegisterName(nameStr);
                r.setOffset(std::stoi(offsetStr, nullptr, 16));
                r.setWidth(std::stoi(rangeStr));
                r.setRW(accessStr);
                r.setDescription(descriptionStr);
                kernel.addRegister(r);
                // std::cout << "Register - Offset: " << offsetStr
                //             << ", Name: " << nameStr
                //             << ", Access: " << accessStr
                //             << ", Description: " << descriptionStr
                //             << ", Range: " << rangeStr << std::endl;
            } else if (nodeNameStr == XML_NODE_VERSION) {
                kernel.setBuildVersion(contentStr);
            } else if (nodeNameStr == XML_NODE_UNIT) {
                kernel.setUnit(contentStr);
            } else if (nodeNameStr == XML_NODE_PRODUCT_FAMILY) {
                kernel.setProductFamily(contentStr);
            } else if (nodeNameStr == XML_NODE_PART) {
                kernel.setPart(contentStr);
            } else if (nodeNameStr == XML_NODE_TOP_MODEL_NAME) {
                kernel.setTopModelName(contentStr);
            } else if (nodeNameStr == XML_NODE_TARGET_CLK) {
                kernel.setTargetClk(std::stof(contentStr));
            } else if (nodeNameStr == XML_NODE_CLK_UNCERTAINTY) {
                kernel.setClkUncertainty(std::stof(contentStr));
            } else if (nodeNameStr == XML_NODE_ESTIMATED_CLK) {
                kernel.setEstimatedClk(std::stof(contentStr));
            } else if (nodeNameStr == XML_NODE_BRAM_18K) {
                Resource resource;
                resource.setName(nodeNameStr);
                resource.setValue(std::stoi(contentStr));
                if (kernel.getAreaEstimates().getUsedResources().empty()) {
                    kernel.addEstimate(RESOURCE_TYPE_USED, resource);
                } else if (kernel.getAreaEstimates().getAvailableResources().empty()) {
                    kernel.addEstimate(RESOURCE_TYPE_AVAILABLE, resource);
                }
            } else if (nodeNameStr == XML_NODE_FF) {
                Resource resource;
                resource.setName(nodeNameStr);
                resource.setValue(std::stoi(contentStr));
                if (kernel.getAreaEstimates().getUsedResources().size() == 1) {
                    kernel.addEstimate(RESOURCE_TYPE_USED, resource);
                } else if (kernel.getAreaEstimates().getAvailableResources().size() == 1) {
                    kernel.addEstimate(RESOURCE_TYPE_AVAILABLE, resource);
                }
            } else if (nodeNameStr == XML_NODE_LUT) {
                Resource resource;
                resource.setName(nodeNameStr);
                resource.setValue(std::stoi(contentStr));
                if (kernel.getAreaEstimates().getUsedResources().size() == 2) {
                    kernel.addEstimate(RESOURCE_TYPE_USED, resource);
                } else if (kernel.getAreaEstimates().getAvailableResources().size() == 2) {
                    kernel.addEstimate(RESOURCE_TYPE_AVAILABLE, resource);
                }
            } else if (nodeNameStr == XML_NODE_URAM) {
                Resource resource;
                resource.setName(nodeNameStr);
                resource.setValue(std::stoi(contentStr));
                if (kernel.getAreaEstimates().getUsedResources().size() == 3) {
                    kernel.addEstimate(RESOURCE_TYPE_USED, resource);
                } else if (kernel.getAreaEstimates().getAvailableResources().size() == 3) {
                    kernel.addEstimate(RESOURCE_TYPE_AVAILABLE, resource);
                }
            } else if (nodeNameStr == XML_NODE_DSP) {
                Resource resource;
                resource.setName(nodeNameStr);
                resource.setValue(std::stoi(contentStr));
                if (kernel.getAreaEstimates().getUsedResources().size() == 4) {
                    kernel.addEstimate(RESOURCE_TYPE_USED, resource);
                } else if (kernel.getAreaEstimates().getAvailableResources().size() == 4) {
                    kernel.addEstimate(RESOURCE_TYPE_AVAILABLE, resource);
                }
            } else if (nodeNameStr == XML_NODE_INTERFACE) {
                Interface intf;
                for (xmlAttr* attr = current_node->properties; attr; attr = attr->next) {
                    xmlChar* value = xmlGetProp(current_node, attr->name);
                    std::string attrNameStr = convertFromXmlCharPtr(attr->name);
                    std::string valueStr = convertFromXmlCharPtr(value);
                    if (attrNameStr == XML_ATTR_INTF_NAME) {
                        intf.setInterfaceName(valueStr);
                    } else if (attrNameStr == XML_ATTR_TYPE) {
                        intf.setInterfaceType(valueStr);
                    } else if (attrNameStr == XML_ATTR_MODE) {
                        intf.setMode(valueStr);
                    } else if (attrNameStr == XML_ATTR_BUS_TYPE) {
                        intf.setBusType(valueStr);
                    } else if (attrNameStr == XML_ATTR_DATA_WIDTH) {
                        intf.setDataWidth(std::stoi(valueStr));
                    } else if (attrNameStr == XML_ATTR_ADDR_WIDTH) {
                        if (std::stoi(valueStr) >= 7) {  // min 2^7 offset
                            intf.setAddrWidth(std::stoi(valueStr));
                        } else {
                            intf.setAddrWidth(7);
                        }
                    }
                    xmlFree(value);
                }
                kernel.addInterface(intf);
            }
            xmlFree(content);
        }
        workingNode = current_node->children;
        parseXml();
    }
}

Kernel XmlParser::getKernel() { return kernel; }