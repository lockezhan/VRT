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

#ifndef DEVICE_HPP
#define DEVICE_HPP

#include <ami.h>
#include <ami_mem_access.h>
#include <ami_program.h>
#include <ami_sensor.h>
#include <fcntl.h>
#include <json/json.h>
#include <libxml/parser.h>
#include <libxml/tree.h>
#include <sys/file.h>
#include <unistd.h>

#include <map>
#include <memory>
#include <thread>

#include "allocator/allocator.hpp"
#include "api/kernel.hpp"
#include "api/vrt_version.hpp"
#include "api/vrtbin.hpp"
#include "driver/clk_wiz.hpp"
#include "driver/qdma_logic.hpp"
#include "parser/xml_parser.hpp"
#include "qdma/pcie_driver_handler.hpp"
#include "qdma/qdma_connection.hpp"
#include "qdma/qdma_intf.hpp"
#include "utils/logger.hpp"
#include "utils/platform.hpp"
#include "utils/zmq_server.hpp"

namespace vrt {

/**
 * @brief Enumeration for device programming types.
 *
 * This enum represents the different methods that can be used to program a device.
 */
enum class ProgramType {
    FLASH,  ///< Program the device using flash memory
    JTAG    ///< Program the device using JTAG interface
};

/**
 * @brief Path to the JTAG programming script.
 *
 * This macro defines the path to the shell script used for programming devices via JTAG.
 */
#define JTAG_PROGRAM_PATH "/usr/local/vrt/jtag_program.sh "

/**
 * @brief Path to the QDMA queue setup script.
 *
 * This macro defines the path to the shell script used for setting up QDMA queues.
 */
#define QDMA_SETUP_QUEUES "/usr/local/vrt/setup_queues.sh "

/**
 * @brief Delay in microseconds for partial boot process.
 *
 * This constant defines the delay time in microseconds that the system
 * will wait during the partial boot process (4 seconds).
 */
#define DELAY_PARTIAL_BOOT (4 * 1000 * 1000)
/**
 * @brief Class representing a device.
 */
class Device {
    static constexpr uint64_t CLK_WIZ_BASE = 0x20100010000;  ///< Base address for the clock wizard
    static constexpr uint32_t CLK_WIZ_OFFSET = 0x10000;
    static constexpr uint64_t QDMA_LOGIC_BASE = 0x20100020000;  ///< Base address for QDMA logic
    static constexpr uint32_t QDMA_LOGIC_OFFSET = 0x1000;       /// Offset for QDMA logic
    ami_device* dev = nullptr;                                  ///< Pointer to the AMI device
    uint8_t bar = 0;                                            ///< Base Address Register (BAR)
    uint64_t offset = 0;                                        ///< Offset for memory operations
    uint16_t pci_bdf = 0;     ///< PCI Bus:Device.Function identifier
    std::string systemMap;    ///< Path to the system map file
    std::string bdf;          ///< Bus:Device.Function identifier
    std::string pdiPath;      ///< Path to the PDI file
    Vrtbin vrtbin;            ///< Vrtbin object for handling VRTBIN operations
    ClkWiz clkWiz;            ///< Clock Wizard object for handling clock wizard operations
    uint64_t clockFreq;       ///< Clock frequency
    ProgramType programType;  ///< Type of programming
    std::map<std::string, Kernel> kernels;        ///< Map of kernel names to Kernel objects
    PcieDriverHandler pcieHandler;                ///< PCIe driver handler object
    Allocator* allocator;                         ///< Allocator object
    VrtbinType vrtbinType;                        ///< Type of VRTBIN
    Platform platform;                            ///< Platform information
    std::shared_ptr<ZmqServer> zmqServer;         ///< ZeroMQ server object
    std::vector<QdmaConnection> qdmaConnections;  ///< Vector of QDMA connections
    std::vector<QdmaIntf*> qdmaIntfs;             ///< Vector of QDMA interfaces for streaming
   public:
    QdmaIntf qdmaIntf;  ///< QDMA interface object

    /**
     * @brief Constructor for Device.
     * @param bdf The Bus:Device.Function identifier.
     * @param vrtbinPath The path to the VRTBIN file.
     * @param program Flag indicating whether to program the device.
     */
    Device(const std::string& bdf, const std::string& vrtbinPath, bool program = true,
           ProgramType programType = ProgramType::FLASH);

    Device() = default;
    /**
     * @brief Gets a kernel by name.
     * @param name The name of the kernel.
     * @return The Kernel object.
     */
    vrt::Kernel getKernel(const std::string& name);

    /**
     * @brief Gets the Bus:Device.Function identifier.
     * @return The Bus:Device.Function identifier.
     */
    std::string getBdf();

    /**
     * @brief Programs the device.
     */
    void programDevice();

    /**
     * @brief Sends a command to the PCIe driver.
     * @param cmd The command to send.
     */
    void sendPcieDriverCmd(std::string cmd);

    /**
     * @brief Boots the device.
     */
    void bootDevice();

    /**
     * @brief Gets a new handle for the device.
     */
    void getNewHandle();

    /**
     * @brief Creates the AMI device.
     */
    void createAmiDev();

    /**
     * @brief Destroys the AMI device.
     */
    void destroyAmiDev();

    /**
     * @brief Destructor for Device.
     */
    ~Device();

    /**
     * @brief Parses the system map file.
     */
    void parseSystemMap();

    /**
     * @brief Cleans up the device.
     */
    void cleanup();
    /**
     * @brief Sets clk_wiz frequency.
     */
    void setFrequency(uint64_t freq);

    /**
     * @brief Gets the clock frequency.
     */
    uint64_t getFrequency();

    /**
     * @brief Gets the maximum frequency.
     */
    uint64_t getMaxFrequency();

    /**
     * @brief Gets ami device.
     */
    ami_device* getAmiDev();

    /**
     * @brief Finds the VRTBIN type from system map.
     */
    void findVrtbinType();

    /**
     * @brief Finds the platform from system map.
     */
    void findPlatform();

    /**
     * @brief Gets the platform.
     */
    Platform getPlatform();

    /**
     * @brief Gets the ZMQ server.
     */
    std::shared_ptr<ZmqServer> getZmqServer();

    /**
     * @brief Gets the Allocator instance.
     */
    Allocator* getAllocator();

    /**
     * @brief Gets the QDMA connections.
     */
    std::vector<QdmaConnection> getQdmaConnections();

    // /**
    //  * @brief Gets the QDMA logic instance.
    //  */
    // QdmaLogic* getQdmaLogic();

    /**
     * @brief Gets the QDMA streaming interfaces.
     */
    std::vector<QdmaIntf*> getQdmaInterfaces();

    /**
     * @brief Locks pcie device, for exclusive access.
     */
    void lockPcieDevice(const std::string& bdf);

    /**
     * @brief Unlocks pcie device, for exclusive access.
     */
    void unlockPcieDevice(const std::string& bdf);
};

}  // namespace vrt

#endif  // DEVICE_HPP