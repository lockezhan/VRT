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

#include "commands/query_command.hpp"

#define SYSTEM_MAP_PATH "%s/%s:00.0/system_map.xml"
#define VERSION_PATH "%s/%s:00.0/version.json"
#define QDMA_QUEUE_PATH "/dev/qdma%s001-MM-0"  // /dev/qdma<bus>:<dev>.<func>-MM-<queue>
#define QDMA_QMAX_PATH "/sys/bus/pci/devices/0000:%s:00.1/qdma/qmax"
#define BUFFER_SIZE 1024

QueryCommand::QueryCommand(const std::string& device) : device(device) {}

void QueryCommand::execute() { queryDevice(); }

void QueryCommand::queryDevice() {
    std::string bdf = device;
    if (bdf.empty()) {
        std::cerr << "Error: BDF is required for query" << std::endl;
        return;
    }

    printAmiDetails();
    queryKernels(bdf);
    queryQueues(bdf);
}

void QueryCommand::queryKernels(const std::string& bdf) {
    std::string bus = bdf;
    char path[256], versionPath[256];
    char* amiHome = getenv("AMI_HOME");
    if (amiHome == nullptr) {
        std::cerr << "Error: AMI_HOME environment variable is not set" << std::endl;
        return;
    }
    sprintf(path, SYSTEM_MAP_PATH, amiHome, bus.c_str());
    sprintf(versionPath, VERSION_PATH, amiHome, bus.c_str());

    xmlDocPtr document = xmlReadFile(path, NULL, 0);
    if (document == NULL) {
        std::cerr << "Error: could not parse file " << path << std::endl;
        return;
    }
    xmlNode* rootNode = xmlDocGetRootElement(document);
    if (rootNode == NULL) {
        std::cerr << "Error: could not get root element" << std::endl;
        return;
    }
    Vrtbin::extractAndPrintInfo(versionPath);
    for (xmlNode* kernelNode = rootNode->children; kernelNode; kernelNode = kernelNode->next) {
        if (kernelNode->type == XML_ELEMENT_NODE &&
            xmlStrcmp(kernelNode->name, BAD_CAST "Kernel") == 0) {
            std::string name, baseAddr, range;
            for (xmlNode* childNode = kernelNode->children; childNode;
                 childNode = childNode->next) {
                if (childNode->type == XML_ELEMENT_NODE) {
                    if (xmlStrcmp(childNode->name, BAD_CAST "Name") == 0) {
                        name = (char*)xmlNodeGetContent(childNode);
                    } else if (xmlStrcmp(childNode->name, BAD_CAST "BaseAddress") == 0) {
                        baseAddr = (char*)xmlNodeGetContent(childNode);
                    } else if (xmlStrcmp(childNode->name, BAD_CAST "Range") == 0) {
                        range = (char*)xmlNodeGetContent(childNode);
                    }
                }
            }
            std::cout << "\t------------------------------------------------------------\n";
            std::cout << "\tKernel Information\n";
            std::cout << "\t------------------------------------------------------------\n";
            std::cout << "\tKernel Name                 | " << name << "\n";
            std::cout << "\tBase Address                | " << baseAddr << "\n";
            std::cout << "\tRange                       | " << range << "\n\n";
        }
    }
    xmlFreeDoc(document);
}

void QueryCommand::queryQueues(const std::string& bdf) {
    std::string bus = bdf;
    char queuePath[256], qmaxPath[256];
    char buffer[BUFFER_SIZE];
    int fd;
    ssize_t bytesRead;

    std::cout << "--------------------------------------------------------------------\n";
    std::cout << "QDMA Queue Status\n";
    std::cout << "--------------------------------------------------------------------\n";
    sprintf(queuePath, QDMA_QUEUE_PATH, bus.c_str());
    sprintf(qmaxPath, QDMA_QMAX_PATH, bus.c_str());

    fd = open(queuePath, O_RDONLY);
    if (fd < 0) {
        std::cerr << "QDMA MM Queue not present. Expected queue " << queuePath << std::endl;
        return;
    } else {
        std::cout << "QDMA MM Queue present at " << queuePath << ", mode bi" << std::endl;
        close(fd);
    }

    fd = open(qmaxPath, O_RDONLY);
    if (fd < 0) {
        std::cerr << "Could not open QMAX file " << qmaxPath << std::endl;
        return;
    } else {
        bytesRead = read(fd, buffer, sizeof(buffer) - 1);
        if (bytesRead < 0) {
            std::cerr << "Error reading from QMAX file " << qmaxPath << std::endl;
            close(fd);
            return;
        } else {
            buffer[bytesRead] = '\0';
            std::cout << "Max allocable queues: " << buffer << std::endl;
        }
        close(fd);
    }
}

void QueryCommand::printAmiDetails() {
    ami_device* dev = nullptr;
    if (ami_dev_find(device.c_str(), &dev) != AMI_STATUS_OK) {
        std::cerr << "Error: AMI Device " << device << " not found" << std::endl;
        return;
    }
    std::cout << "Query device: " << device << ":00.0" << std::endl;
    std::cout << "--------------------------------------------------------------------\n";
    std::cout << "AMI Device Information\n";
    std::cout << "--------------------------------------------------------------------\n";

    char devName[AMI_DEV_NAME_SIZE];

    if (ami_dev_get_name(dev, devName) != AMI_STATUS_OK) {
        std::cerr << "Error: Failed to get device name" << std::endl;
        return;
    }

    char devState[AMI_DEV_STATE_SIZE];

    if (ami_dev_get_state(dev, devState) != AMI_STATUS_OK) {
        std::cerr << "Error: Failed to get device state" << std::endl;
        return;
    }

    char uuid[AMI_LOGIC_UUID_SIZE];
    if (ami_dev_read_uuid(dev, uuid) != AMI_STATUS_OK) {
        std::cerr << "Error: Failed to read UUID" << std::endl;
        return;
    }

    struct ami_version ami_version;

    if (ami_get_driver_version(&ami_version) != AMI_STATUS_OK) {
        std::cerr << "Error: Failed to get driver version" << std::endl;
        return;
    }

    struct ami_version api_version;

    if (ami_get_api_version(&api_version) != AMI_STATUS_OK) {
        std::cerr << "Error: Failed to get API version" << std::endl;
        return;
    }

    struct amc_version amc_version;

    if (ami_dev_get_amc_version(dev, &amc_version) != AMI_STATUS_OK) {
        std::cerr << "Error: Failed to get AMC version" << std::endl;
        return;
    }

    std::cout << "Device Name                 | " << devName << "\n";
    std::cout << "Device State                | " << devState << "\n";
    std::cout << "Logic UUID                  | " << std::string(uuid).substr(0, 32) << "\n";
    std::cout << "Driver Version              | " << std::to_string(ami_version.major) << "."
              << std::to_string(ami_version.minor) << "." << std::to_string(ami_version.patch)
              << "\n";
    std::cout << "API Version                 | " << std::to_string(api_version.major) << "."
              << std::to_string(api_version.minor) << "." << std::to_string(api_version.patch)
              << "\n";
    std::cout << "FW Version                  | " << std::to_string(amc_version.major) << "."
              << std::to_string(amc_version.minor) << "." << std::to_string(amc_version.patch)
              << "\n";
    std::cout << "\n";

    uint16_t pciVendor;
    if (ami_dev_get_pci_vendor(dev, &pciVendor) != AMI_STATUS_OK) {
        std::cerr << "Error: Failed to get PCI vendor" << std::endl;
        return;
    }

    uint16_t pciDevice;
    if (ami_dev_get_pci_device(dev, &pciDevice) != AMI_STATUS_OK) {
        std::cerr << "Error: Failed to get PCI device" << std::endl;
        return;
    }

    uint8_t currentLinkSpeed, maxLinkSpeed;
    if (ami_dev_get_pci_link_speed(dev, &currentLinkSpeed, &maxLinkSpeed) != AMI_STATUS_OK) {
        std::cerr << "Error: Failed to get PCI link speed" << std::endl;
        return;
    }

    uint8_t currentLinkWidth, maxLinkWidth;
    if (ami_dev_get_pci_link_width(dev, &currentLinkWidth, &maxLinkWidth) != AMI_STATUS_OK) {
        std::cerr << "Error: Failed to get PCI link width" << std::endl;
        return;
    }

    uint8_t numaNode;

    if (ami_dev_get_pci_numa_node(dev, &numaNode) != AMI_STATUS_OK) {
        std::cerr << "Error: Failed to get PCI NUMA mode" << std::endl;
        return;
    }

    char cpuList[AMI_PCI_CPULIST_SIZE];
    if (ami_dev_get_pci_cpulist(dev, cpuList) != AMI_STATUS_OK) {
        std::cerr << "Error: Failed to get CPU affinity" << std::endl;
        return;
    }

    char productName[AMI_MFG_INFO_MAX_STR], boardRev[AMI_MFG_INFO_MAX_STR],
        eepromVersion[AMI_MFG_INFO_MAX_STR], boardSerial[AMI_MFG_INFO_MAX_STR],
        partNum[AMI_MFG_INFO_MAX_STR], mPartNum[AMI_MFG_INFO_MAX_STR],
        macAddr[AMI_MFG_INFO_MAX_STR], macAddrN[AMI_MFG_INFO_MAX_STR], mDate[AMI_MFG_INFO_MAX_STR],
        uuid_system[AMI_MFG_INFO_MAX_STR];

    if (ami_mfg_get_info(dev, AMI_MFG_PRODUCT_NAME, productName) != AMI_STATUS_OK) {
        std::cerr << "Error: Failed to get product name" << std::endl;
        return;
    }

    if (ami_mfg_get_info(dev, AMI_MFG_BOARD_REV, boardRev) != AMI_STATUS_OK) {
        std::cerr << "Error: Failed to get board revision" << std::endl;
        return;
    }

    if (ami_mfg_get_info(dev, AMI_MFG_EEPROM_VERSION, eepromVersion) != AMI_STATUS_OK) {
        std::cerr << "Error: Failed to get eeprom version" << std::endl;
        return;
    }
    if (ami_mfg_get_info(dev, AMI_MFG_BOARD_SERIAL, boardSerial) != AMI_STATUS_OK) {
        std::cerr << "Error: Failed to get board serial" << std::endl;
        return;
    }
    if (ami_mfg_get_info(dev, AMI_MFG_PART_NUM, partNum) != AMI_STATUS_OK) {
        std::cerr << "Error: Failed to get part number" << std::endl;
        return;
    }
    if (ami_mfg_get_info(dev, AMI_MFG_M_PART_NUM, mPartNum) != AMI_STATUS_OK) {
        std::cerr << "Error: Failed to get m part number" << std::endl;
        return;
    }

    if (ami_mfg_get_info(dev, AMI_MFG_MAC_ADDR, macAddr) != AMI_STATUS_OK) {
        std::cerr << "Error: Failed to get mac addr" << std::endl;
        return;
    }

    if (ami_mfg_get_info(dev, AMI_MFG_MAC_ADDR_N, macAddrN) != AMI_STATUS_OK) {
        std::cerr << "Error: Failed to get mac addr n" << std::endl;
        return;
    }

    if (ami_mfg_get_info(dev, AMI_MFG_M_DATE, mDate) != AMI_STATUS_OK) {
        std::cerr << "Error: Failed to get m date" << std::endl;
        return;
    }

    if (ami_mfg_get_info(dev, AMI_MFG_UUID, uuid_system) != AMI_STATUS_OK) {
        std::cerr << "Error: Failed to get uuid" << std::endl;
        return;
    }

    std::cout << "--------------------------------------------------------------------\n";
    std::cout << "Manufacturing Information\n";
    std::cout << "--------------------------------------------------------------------\n";
    std::cout << "Product Name                | " << productName << "\n";
    std::cout << "Board Revision              | " << boardRev << "\n";
    std::cout << "EEPROM Version              | " << eepromVersion << "\n";
    std::cout << "Board Serial Number         | " << boardSerial << "\n";
    std::cout << "Part Number                 | " << partNum << "\n";
    std::cout << "M Part Number               | " << mPartNum << "\n";
    std::cout << "MAC Address                 | " << macAddr << "\n";
    std::cout << "MAC Address N               | " << macAddrN << "\n";
    formatManufacturingDate(std::strtoul(mDate, nullptr, 10));
    std::cout << "UUID                        | " << uuid_system << "\n\n";

    std::cout << "--------------------------------------------------------------------\n";
    std::cout << "PCI Information\n";
    std::cout << "--------------------------------------------------------------------\n";
    std::cout << "PCI Vendor                  | " << std::hex << std::showbase << pciVendor
              << std::dec << "\n";
    std::cout << "PCI Device                  | " << std::hex << std::showbase << pciDevice
              << std::dec << "\n";
    std::cout << "Current Link Speed          | Gen" << (int)currentLinkSpeed << " (max Gen"
              << (int)maxLinkSpeed << ")\n";
    std::cout << "Current Link Width          | x" << (int)currentLinkWidth << " (max x"
              << (int)maxLinkWidth << ")\n";
    std::cout << "NUMA Mode                   | " << (int)numaNode << "\n";
    std::cout << "CPU Affinity                | " << cpuList << "\n\n";
}

void QueryCommand::formatManufacturingDate(long manufacturing_date_mins) {
    char manufacturing_date_str[META_MAX_STR_LEN] = {0};
    struct tm info = {0};

    if (manufacturing_date_mins) {
        info.tm_year = 96;  // Base year 1970
        info.tm_mon = 1;
        info.tm_mday = 1;
        info.tm_hour = 0;
        info.tm_min = manufacturing_date_mins;
        info.tm_sec = 0;
        info.tm_isdst = -1;

        if (mktime(&info) == -1) {
            std::cerr << "Error: mktime failed" << std::endl;
            return;
        }

        if (!strftime(manufacturing_date_str, sizeof(manufacturing_date_str), "%c", &info)) {
            std::cerr << "Error: strftime failed" << std::endl;
            return;
        }
    } else {
        std::cerr << "Error: Invalid manufacturing date minutes" << std::endl;
        return;
    }

    std::cout << "MFG Date                    | " << manufacturing_date_str << std::endl;
}
