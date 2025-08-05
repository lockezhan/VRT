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

#include "system_map.hpp"

SystemMap::SystemMap(Platform platform) {
    this->platform = platform;
}

void SystemMap::addEntry(MapEntry entry) { this->entries.emplace_back(entry); }

std::string SystemMap::intToHex(uint64_t value) {
    std::stringstream ss;
    ss << std::hex << std::showbase << value;
    return ss.str();
}

// TODO: Prints to xml file
void SystemMap::printToFile() {
    xmlDocPtr doc = xmlNewDoc(BAD_CAST "1.0");
    xmlNodePtr rootNode = xmlNewNode(NULL, BAD_CAST "SystemMap");
    xmlDocSetRootElement(doc, rootNode);
    xmlNewChild(rootNode, NULL, BAD_CAST "Platform",
                BAD_CAST(platform == Platform::HARDWARE
                             ? "Hardware"
                             : (platform == Platform::EMULATOR ? "Emulation" : "Simulation")));
    xmlNewChild(rootNode, NULL, BAD_CAST "Type", BAD_CAST "Full");
    xmlNewChild(rootNode, NULL, BAD_CAST "ClockFrequency",
                BAD_CAST std::to_string(targetClockFreq).c_str());
    for (auto& entry : entries) {
        xmlNodePtr newNode = xmlNewChild(rootNode, NULL, BAD_CAST "Kernel", NULL);
        xmlNewChild(newNode, NULL, BAD_CAST "Name", BAD_CAST entry.getName().c_str());
        xmlNewChild(newNode, NULL, BAD_CAST "BaseAddress",
                    BAD_CAST intToHex(entry.getBaseAddr()).c_str());
        xmlNewChild(newNode, NULL, BAD_CAST "Range", BAD_CAST intToHex(entry.getRange()).c_str());
        for (auto& reg : entry.getRegisters()) {
            xmlNodePtr register_node = xmlNewChild(newNode, NULL, BAD_CAST "register", NULL);
            xmlNewProp(register_node, BAD_CAST "offset",
                       BAD_CAST intToHex(reg.getOffset()).c_str());
            xmlNewProp(register_node, BAD_CAST "name", BAD_CAST reg.getRegisterName().c_str());
            xmlNewProp(register_node, BAD_CAST "access", BAD_CAST reg.getRW().c_str());
            xmlNewProp(register_node, BAD_CAST "description",
                       BAD_CAST reg.getDescription().c_str());
            xmlNewProp(register_node, BAD_CAST "range",
                       BAD_CAST std::to_string(reg.getWidth()).c_str());
        }
    }

    for (auto& sc : qdmaStreamConnections) {
        xmlNodePtr newNode = xmlNewChild(rootNode, NULL, BAD_CAST "Qdma", NULL);
        xmlNewChild(newNode, NULL, BAD_CAST "kernel", BAD_CAST sc.kernelName.c_str());
        xmlNewChild(newNode, NULL, BAD_CAST "interface", BAD_CAST sc.interfaceName.c_str());
        xmlNewChild(newNode, NULL, BAD_CAST "qid", BAD_CAST std::to_string(sc.qid).c_str());
        xmlNewChild(newNode, NULL, BAD_CAST "direction",
                    BAD_CAST(sc.direction == StreamDirection::DEVICE_TO_HOST ? "DeviceToHost"
                                                                             : "HostToDevice"));
    }

    xmlSaveFormatFileEnc("system_map.xml", doc, "UTF-8", 1);
    xmlFreeDoc(doc);
    xmlCleanupParser();

    // std::ofstream systemMapOutputFile(SYSTEM_MAP_OUTPUT);
    // systemMapOutputFile << "# System Map\n";
    // systemMapOutputFile << "# Name\tBase Address\tRange\n";
    // for (auto& entry : entries) {
    //     systemMapOutputFile << "  "<< entry.getName() << '\t' << std::hex << std::showbase <<
    //     entry.getBaseAddr() << "\t" << entry.getRange() << std::endl;
    // }
    // systemMapOutputFile <<"# System Map end\n";
    // systemMapOutputFile.close();
}

void SystemMap::setClockFreq(uint64_t freq) { this->targetClockFreq = freq; }

void SystemMap::addStreamConnection(StreamingConnection connection) {
    this->qdmaStreamConnections.emplace_back(connection);
}