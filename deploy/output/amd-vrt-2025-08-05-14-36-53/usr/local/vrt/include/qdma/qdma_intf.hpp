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

#ifndef QDMA_INTF_HPP
#define QDMA_INTF_HPP

#include <errno.h>
#include <fcntl.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include <string>

#include "utils/logger.hpp"

#define RW_MAX_SIZE 0x7ffff000  ///< Maximum size for read/write operations
#define GB_DIV 1000000000       ///< Divider for gigabytes
#define MB_DIV 1000000          ///< Divider for megabytes
#define KB_DIV 1000             ///< Divider for kilobytes
#define NSEC_DIV 1000000000     ///< Divider for nanoseconds
#define QMAX_PATH "/sys/bus/pci/devices/0000:%s:00.1/qdma/qmax"  ///< Path for QMAX
#define QDMA_QUEUE_NAME "qdma%s001"                              ///< Format for QDMA queue name
#define QDMA_DEFAULT_QUEUE "/dev/qdma%s001-MM-0"                 ///< Default QDMA queue
#define QDMA_DEFAULT_ST_QUEUE "/dev/qdma%s001-ST-%u"             ///< Default stream queue

namespace vrt {
/**
 * @brief Class for interfacing with QDMA.
 */
class QdmaIntf {
    uint8_t queueIdx;       ///< Queue index
    std::string bdf;        ///< Bus:Device.Function identifier
    std::string queueName;  ///< Queue name

    /**
     * @brief Writes data from a buffer to a device.
     * @param dev The device to write to.
     * @param buffer The buffer to write from.
     * @param size The size of the buffer.
     * @param base The base address to write to.
     * @return The number of bytes written.
     */
    ssize_t write_from_buffer(const char* dev, char* buffer, uint64_t size, uint64_t base);

    /**
     * @brief Reads data from a file into a buffer.
     * @param fname The file to read from.
     * @param buffer The buffer to read into.
     * @param size The size of the buffer.
     * @param base The base address to read from.
     * @return The number of bytes read.
     */
    ssize_t read_to_buffer(const char* fname, char* buffer, uint64_t size, uint64_t base);

    /**
     * @brief Strips the bus part from the BDF.
     * @param bdf The BDF to strip.
     * @return The stripped bus part.
     */
    char* strip(const char* bdf);

    /**
     * @brief Creates a QDMA queue.
     * @param bdf The BDF of the device.
     * @return The name of the created queue.
     */
    char* create_qdma_queue(const char* bdf);

    /**
     * @brief Deletes a QDMA queue.
     * @param bdf The BDF of the device.
     * @return 0 on success, -1 on failure.
     */
    int delete_qdma_queue(const char* bdf);

    // Static instance pointer
    static QdmaIntf* instance;

   public:
    /**
     * @brief Constructor of the QdmaIntf class
     * @param bdf The BDF (Bus:Device.Function) of the device.
     */
    QdmaIntf(const std::string& bdf);

    /**
     * @brief Constructor of the QdmaIntf class with queue index
     * @brief Assumes all queues are stream type
     * @param bdf The BDF (Bus:Device.Function) of the device.
     * @param queueIdx The index of the queue.
     */
    QdmaIntf(const std::string& bdf, const uint32_t queueIdx);

    /**
     * @brief Default constructor for QdmaIntf.
     */
    QdmaIntf() = default;

    /**
     * @brief Writes a buffer to the device.
     * @param buffer The buffer to write.
     * @param start_addr The starting address to write to.
     * @param size The size of the buffer.
     */
    void write_buff(char* buffer, uint64_t start_addr, uint64_t size);

    /**
     * @brief Reads a buffer from the device.
     * @param buffer The buffer to read into.
     * @param start_addr The starting address to read from.
     * @param size The size of the buffer.
     */
    void read_buff(char* buffer, uint64_t start_addr, uint64_t size);

    /**
     * @brief Gets the queue index.
     * @return The queue index.
     */
    uint32_t getQueueIdx();
    /**
     * @brief Destructor for QdmaIntf.
     */
    ~QdmaIntf();
};

}  // namespace vrt
#endif  // QDMA_INTF_HPP