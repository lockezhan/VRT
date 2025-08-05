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

#ifndef VALIDATE_CMD_HPP
#define VALIDATE_CMD_HPP

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <string>
#include <thread>
#include <vector>

/**
 * @brief Starting address for DDR memory.
 *
 * This macro defines the starting address for DDR memory testing.
 */
#define ADDR_START_DDR 0x50080000000

/**
 * @brief Starting address for DIMM DDR memory.
 *
 * This macro defines the starting address for DIMM DDR memory testing.
 */
#define ADDR_START_DIMM_DDR 0x60000000000

/**
 * @brief Starting address for HBM memory.
 *
 * This macro defines the starting address for HBM memory testing.
 */
#define ADDR_START_HBM 0x4000000000

/**
 * @brief Size per HBM channel.
 *
 * This macro defines the size of each HBM channel.
 */
#define SIZE_PER_HBM_CHANNEL 0x40000000

/**
 * @brief Default test size in bytes.
 *
 * This macro defines the default size used for memory tests.
 */
#define DEFAULT_TEST_SIZE 255000000

/**
 * @brief Default number of test iterations.
 *
 * This macro defines the default number of times each test is run.
 */
#define DEFAULT_COUNT_TIMES 50

/**
 * @brief Maximum size for read/write operations.
 *
 * This macro defines the maximum allowed size for a single read/write operation.
 */
#define RW_MAX_SIZE 0x7ffff000

/**
 * @brief Divisor for gigabyte conversion.
 *
 * This macro defines the divisor used to convert bytes to gigabytes.
 */
#define GB_DIV 1000000000

/**
 * @brief Divisor for megabyte conversion.
 *
 * This macro defines the divisor used to convert bytes to megabytes.
 */
#define MB_DIV 1000000

/**
 * @brief Divisor for kilobyte conversion.
 *
 * This macro defines the divisor used to convert bytes to kilobytes.
 */
#define KB_DIV 1000

/**
 * @brief Divisor for nanosecond conversion.
 *
 * This macro defines the divisor used to convert nanoseconds to seconds.
 */
#define NSEC_DIV 1000000000

/**
 * @brief Enumeration for test types.
 */
typedef enum {
    TEST_TYPE_READ = 0, /**< Read test type. */
    TEST_TYPE_WRITE     /**< Write test type. */
} test_type;

/**
 * @brief Structure to hold thread data.
 */
typedef struct {
    std::string devname; /**< Device name. */
    uint64_t addr;       /**< Address to test. */
    uint64_t size;       /**< Size of the memory region. */
    uint64_t offset;     /**< Offset within the memory region. */
    uint64_t count;      /**< Number of iterations. */
    int verbose;         /**< Verbose flag. */
} thread_data_t;

/**
 * @brief Class for validating device memory and bandwidth.
 *
 * The ValidateCommand class provides functionality to validate the memory and
 * bandwidth of a device by running various DMA read and write tests.
 */
class ValidateCommand {
   public:
    /**
     * @brief Constructor for ValidateCommand.
     *
     * @param device The BDF of the device to validate.
     */
    ValidateCommand(const std::string& device);

    /**
     * @brief Executes the validate command.
     *
     * This method runs a series of memory and bandwidth tests on the specified device.
     */
    void execute();

   private:
    std::string device;  ///< The BDF of the device to validate.

    /**
     * @brief Performs a DMA test.
     *
     * @param devname The device name.
     * @param addr The memory address to test.
     * @param size The size of the memory region to test.
     * @param offset The offset within the memory region.
     * @param count The number of test iterations.
     * @param verbose Flag for verbose output.
     *
     * This method performs a DMA test with the specified parameters.
     */
    void test_dma(const std::string& devname, uint64_t addr, uint64_t size, uint64_t offset,
                  uint64_t count, int verbose);

    /**
     * @brief Writes data from a buffer to a device.
     *
     * @param fname The device file name.
     * @param fd The file descriptor.
     * @param buffer The buffer containing data to write.
     * @param size The size of data to write.
     * @param base The base address to write to.
     * @return The number of bytes written.
     *
     * This method writes data from a buffer to a device at the specified address.
     */
    std::size_t write_from_buffer(const std::string& fname, int fd, char* buffer, uint64_t size,
                                  uint64_t base);

    /**
     * @brief Reads data from a device to a buffer.
     *
     * @param fname The device file name.
     * @param fd The file descriptor.
     * @param buffer The buffer to read data into.
     * @param size The size of data to read.
     * @param base The base address to read from.
     * @return The number of bytes read.
     *
     * This method reads data from a device at the specified address into a buffer.
     */
    std::size_t read_to_buffer(const std::string& fname, int fd, char* buffer, uint64_t size,
                               uint64_t base);

    /**
     * @brief Tests DMA write operations.
     *
     * @param devname The device name.
     * @param addr The memory address to test.
     * @param size The size of the memory region to test.
     * @param offset The offset within the memory region.
     * @param count The number of test iterations.
     * @param verbose Flag for verbose output.
     * @return The number of bytes written.
     *
     * This method tests DMA write operations with the specified parameters.
     */
    std::size_t test_dma_write(const std::string& devname, uint64_t addr, uint64_t size,
                               uint64_t offset, uint64_t count, int verbose);

    /**
     * @brief Tests DMA read operations.
     *
     * @param devname The device name.
     * @param addr The memory address to test.
     * @param size The size of the memory region to test.
     * @param offset The offset within the memory region.
     * @param count The number of test iterations.
     * @param verbose Flag for verbose output.
     * @return The number of bytes read.
     *
     * This method tests DMA read operations with the specified parameters.
     */
    std::size_t test_dma_read(const std::string& devname, uint64_t addr, uint64_t size,
                              uint64_t offset, uint64_t count, int verbose);

    /**
     * @brief Prints test results.
     *
     * @param devname The device name.
     * @param addr The memory address tested.
     * @param total_time The total time taken for the test.
     * @param avg_time The average time per iteration.
     * @param size The size of the memory region tested.
     * @param result The test result.
     * @param type The type of test (read or write).
     * @param verbose Flag for verbose output.
     *
     * This method prints the results of a DMA test.
     */
    void print_results(const std::string& devname, uint64_t addr, double total_time,
                       double avg_time, double size, double result, test_type type, int verbose);

    /**
     * @brief Prints PCIe bandwidth results.
     *
     * @param pci_bw_result The PCIe bandwidth measurement result.
     *
     * This method prints the results of a PCIe bandwidth test.
     */
    void print_pci_bandwidth(double pci_bw_result);

    /**
     * @brief Thread function for DMA read tests.
     *
     * @param data Pointer to thread data structure.
     *
     * This method is executed in a thread to perform DMA read tests.
     */
    void dma_read_thread(thread_data_t* data);

    /**
     * @brief Thread function for DMA write tests.
     *
     * @param data Pointer to thread data structure.
     *
     * This method is executed in a thread to perform DMA write tests.
     */
    void dma_write_thread(thread_data_t* data);

    /**
     * @brief Runs sequential read/write tests in threads.
     *
     * @param devname The device name.
     * @param size The size of the memory region to test.
     * @param offset The offset within the memory region.
     * @param count The number of test iterations.
     * @param verbose Flag for verbose output.
     * @return The result code.
     *
     * This method runs sequential read/write tests in separate threads.
     */
    int run_sim_seq_rw_threads(const std::string& devname, uint64_t size, uint64_t offset,
                               uint64_t count, int verbose);

    /**
     * @brief Runs read/write tests for a specific memory type.
     *
     * @param devname The device name.
     * @param addr The memory address to test.
     * @param size The size of the memory region to test.
     * @param offset The offset within the memory region.
     * @param count The number of test iterations.
     * @param verbose Flag for verbose output.
     * @param mem_type The type of memory being tested.
     * @return The result code.
     *
     * This method runs read/write tests for a specific memory type.
     */
    int run_sim_rw_per_memory(const std::string& devname, uint64_t addr, uint64_t size,
                              uint64_t offset, uint64_t count, int verbose,
                              const std::string& mem_type);

    /**
     * @brief Runs PCIe bandwidth tests.
     *
     * @param devname The device name.
     * @param size The size of the memory region to test.
     * @param offset The offset within the memory region.
     * @param count The number of test iterations.
     * @param verbose Flag for verbose output.
     * @return The result code.
     *
     * This method runs PCIe bandwidth tests.
     */
    int run_pcie_bw_test(const std::string& devname, uint64_t size, uint64_t offset, uint64_t count,
                         int verbose);

    /**
     * @brief Tests DMA write operations for PCIe bandwidth measurement.
     *
     * @param devname The device name.
     * @param addr The memory address to test.
     * @param size The size of the memory region to test.
     * @param offset The offset within the memory region.
     * @param count The number of test iterations.
     * @param verbose Flag for verbose output.
     * @return The bandwidth measurement result.
     *
     * This method tests DMA write operations for PCIe bandwidth measurement.
     */
    double test_dma_write_pcie(const std::string& devname, uint64_t addr, uint64_t size,
                               uint64_t offset, uint64_t count, int verbose);

    /**
     * @brief Thread function for PCIe DMA write tests.
     *
     * @param data Pointer to thread data structure.
     *
     * This method is executed in a thread to perform PCIe DMA write tests.
     */
    void dma_write_thread_pcie(thread_data_t* data);

    /**
     * @brief Subtracts one timespec structure from another.
     *
     * @param t1 Pointer to the first timespec structure.
     * @param t2 Pointer to the second timespec structure.
     *
     * This method subtracts t2 from t1, storing the result in t1.
     */
    void timespec_sub(struct timespec* t1, struct timespec* t2);

    /**
     * @brief Checks if a timespec structure is valid.
     *
     * @param t Pointer to the timespec structure to check.
     * @return 1 if valid, 0 otherwise.
     *
     * This method checks if a timespec structure is valid.
     */
    int timespec_check(struct timespec* t);
};

#endif  // VALIDATE_CMD_HPP