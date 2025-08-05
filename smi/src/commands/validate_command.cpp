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

#include "commands/validate_command.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

static std::atomic<double> pci_bw_result(0);

ValidateCommand::ValidateCommand(const std::string& device) : device(device) {}

void ValidateCommand::execute() {
    if (device.empty()) {
        std::cerr << "Error: Device is required for validation" << std::endl;
        return;
    }

    std::cout << "Running validation for device: " << device << std::endl;

    uint64_t address = 0;
    uint64_t size = DEFAULT_TEST_SIZE;
    uint64_t offset = 0;
    uint64_t count = DEFAULT_COUNT_TIMES;
    int verbose = 0;
    int read = 0;
    int write = 0;
    int port = 0;
    char cmd[256];
    sprintf(cmd, "/dev/qdma%s001-MM-0", device.c_str());
    if (access(cmd, F_OK) != 0) {
        std::cerr << "Error: Device " << cmd << " does not exist" << std::endl;
        std::cerr << "Please run as root: /usr/local/vrt/setup_queues.sh " << cmd << " --mm 0 bi\n";
        return;
    }
    test_dma(std::string(cmd), address, size, offset, count, verbose);

    std::cout << "All tests passed" << std::endl;
}

void ValidateCommand::test_dma(const std::string& devname, uint64_t addr, uint64_t size,
                               uint64_t offset, uint64_t count, int verbose) {
    int ret = -1;
    addr = ADDR_START_HBM;
    std::cout << "Performing seq RW test for HBM\n";
    ret = test_dma_write(devname, addr, size, offset, count, verbose);
    if (ret < 0) {
        std::cerr << "Error: Write test failed" << std::endl;
        return;
    }
    ret = test_dma_read(devname, addr, size, offset, count, verbose);
    if (ret < 0) {
        std::cerr << "Error: Read test failed" << std::endl;
        return;
    }
    addr = ADDR_START_DDR;
    std::cout << "Performing seq RW test for DDR\n";
    ret = test_dma_write(devname, addr, size, offset, count, verbose);
    if (ret < 0) {
        std::cerr << "Error: Write test failed" << std::endl;
        return;
    }
    ret = test_dma_read(devname, addr, size, offset, count, verbose);
    if (ret < 0) {
        std::cerr << "Error: Read test failed" << std::endl;
        return;
    }
    ret = run_sim_seq_rw_threads(devname, size, offset, count, verbose);
    if (ret < 0) {
        std::cerr << "Error: Simultaneous RW test failed" << std::endl;
        return;
    }
    ret = run_sim_rw_per_memory(devname, addr, size, offset, count, verbose, "hbm");
    if (ret < 0) {
        std::cerr << "Error: Simultaneous RW test per memory failed" << std::endl;
        return;
    }
    ret = run_sim_rw_per_memory(devname, addr, size, offset, count, verbose, "ddr");
    if (ret < 0) {
        std::cerr << "Error: Simultaneous RW test per memory failed" << std::endl;
        return;
    }
    ret = run_pcie_bw_test(devname, size, offset, count, verbose);
    if (ret < 0) {
        std::cerr << "Error: PCIe bandwidth test failed" << std::endl;
        return;
    }
}

std::size_t ValidateCommand::test_dma_write(const std::string& devname, uint64_t addr,
                                            uint64_t size, uint64_t offset, uint64_t count,
                                            int verbose) {
    uint64_t i;
    char* buffer = NULL;
    char* allocated = NULL;
    struct timespec ts_start, ts_end;
    int fpga_fd = open(devname.c_str(), O_RDWR);
    double total_time = 0;
    double result;
    double avg_time = 0;
    int ret = EXIT_FAILURE;
    if (fpga_fd < 0) {
        fprintf(stderr, "unable to open device %s, %d.\n", devname.c_str(), fpga_fd);
        return EXIT_FAILURE;
    }
    posix_memalign((void**)&allocated, 4096 /*alignment */, size + 4096);
    if (!allocated) {
        fprintf(stderr, "OOM %lu.\n", size + 4096);
        close(fpga_fd);
        free(allocated);
        return EXIT_FAILURE;
    }
    buffer = allocated + offset;
    if (verbose) {
        fprintf(stdout, "host buffer 0x%lx = %p\n", size + 4096, buffer);
    }
    for (i = 0; i < count; i++) {
        clock_gettime(CLOCK_MONOTONIC, &ts_start);
        ret = write_from_buffer(devname, fpga_fd, buffer, size, addr);
        if (ret < 0) {
            fprintf(stderr, "Could not write to device buffer.\n");
            close(fpga_fd);
            free(allocated);
            return EXIT_FAILURE;
        }
        clock_gettime(CLOCK_MONOTONIC, &ts_end);
        timespec_sub(&ts_end, &ts_start);
        total_time += (ts_end.tv_sec + ((double)ts_end.tv_nsec / NSEC_DIV));
        if (verbose) {
            fprintf(stdout, "CLOCK_MONOTONIC %ld.%09ld sec. write %lu bytes\n", ts_end.tv_sec,
                    ts_end.tv_nsec, size);
        }
    }
    avg_time = (double)total_time / (double)count;
    result = ((double)size) / avg_time;
    // pthread_mutex_lock(&print_mutex);
    print_results(devname, addr, total_time, avg_time, size, result, TEST_TYPE_WRITE, verbose);
    // pthread_mutex_unlock(&print_mutex);
    close(fpga_fd);
    free(allocated);
    return 0;
}

std::size_t ValidateCommand::test_dma_read(const std::string& devname, uint64_t addr, uint64_t size,
                                           uint64_t offset, uint64_t count, int verbose) {
    uint64_t i;
    char* buffer = NULL;
    char* allocated = NULL;
    struct timespec ts_start, ts_end;
    int fpga_fd = open(devname.c_str(), O_RDWR | O_NONBLOCK);
    double total_time = 0;
    double result;
    double avg_time = 0;
    int ret = EXIT_FAILURE;
    if (fpga_fd < 0) {
        fprintf(stderr, "unable to open device %s, %d.\n", devname.c_str(), fpga_fd);
        return 1;
    }
    posix_memalign((void**)&allocated, 4096 /*alignment */, size + 4096);
    if (!allocated) {
        fprintf(stderr, "OOM %lu.\n", size + 4096);
        close(fpga_fd);
        free(allocated);
        return EXIT_FAILURE;
    }
    buffer = allocated + offset;
    if (verbose) fprintf(stdout, "host buffer 0x%lx = %p\n", size + 4096, buffer);

    for (i = 0; i < count; i++) {
        clock_gettime(CLOCK_MONOTONIC, &ts_start);
        /* lseek & read data from AXI MM into buffer using SGDMA */
        ret = read_to_buffer(devname, fpga_fd, buffer, size, addr);
        if (ret < 0) {
            fprintf(stderr, "Could not read to device buffer.\n");
            close(fpga_fd);
            free(allocated);
            return 1;
        }
        clock_gettime(CLOCK_MONOTONIC, &ts_end);

        /* subtract the start time from the end time */
        timespec_sub(&ts_end, &ts_start);
        total_time += (ts_end.tv_sec + ((double)ts_end.tv_nsec / NSEC_DIV));
        /* a bit less accurate but side-effects are accounted for */
        if (verbose) {
            fprintf(stdout, "#%lu: CLOCK_MONOTONIC %ld.%09ld sec. read %lu bytes\n", i,
                    ts_end.tv_sec, ts_end.tv_nsec, size);
        }
    }
    avg_time = (double)total_time / (double)count;
    result = ((double)size) / avg_time;
    // pthread_mutex_lock(&print_mutex);
    this->print_results(devname, addr, total_time, avg_time, size, result, TEST_TYPE_READ, verbose);
    // pthread_mutex_unlock(&print_mutex);
    close(fpga_fd);
    free(allocated);
    return 0;
}

std::size_t ValidateCommand::write_from_buffer(const std::string& fname, int fd, char* buffer,
                                               uint64_t size, uint64_t base) {
    ssize_t rc;
    uint64_t count = 0;
    char* buf = buffer;
    off_t offset = base;

    do { /* Support zero byte transfer */
        uint64_t bytes = size - count;

        if (bytes > RW_MAX_SIZE) bytes = RW_MAX_SIZE;

        if (offset) {
            rc = lseek(fd, offset, SEEK_SET);
            if (rc < 0) {
                fprintf(stderr, "%s, seek off 0x%lx failed %zd.\n", fname.c_str(), offset, rc);
                perror("seek file");
                return -EIO;
            }
            if (rc != offset) {
                fprintf(stderr, "%s, seek off 0x%lx != 0x%lx.\n", fname.c_str(), rc, offset);
                return -EIO;
            }
        }

        /* write data to file from memory buffer */
        rc = write(fd, buf, bytes);
        if (rc < 0) {
            fprintf(stderr, "%s, W off 0x%lx, 0x%lx failed %zd.\n", fname.c_str(), offset, bytes,
                    rc);
            perror("write file");
            return -EIO;
        }
        if (rc != bytes) {
            fprintf(stderr, "%s, W off 0x%lx, 0x%lx != 0x%lx.\n", fname.c_str(), offset, rc, bytes);
            return -EIO;
        }

        count += bytes;
        buf += bytes;
        offset += bytes;
    } while (count < size);

    if (count != size) {
        fprintf(stderr, "%s, R failed 0x%lx != 0x%lx.\n", fname.c_str(), count, size);
        return -EIO;
    }
    return count;
}

std::size_t ValidateCommand::read_to_buffer(const std::string& fname, int fd, char* buffer,
                                            uint64_t size, uint64_t base) {
    ssize_t rc;
    uint64_t count = 0;
    char* buf = buffer;
    off_t offset = base;

    do { /* Support zero byte transfer */
        uint64_t bytes = size - count;

        if (bytes > RW_MAX_SIZE) bytes = RW_MAX_SIZE;

        if (offset) {
            rc = lseek(fd, offset, SEEK_SET);
            if (rc < 0) {
                fprintf(stderr, "%s, seek off 0x%lx failed %zd.\n", fname.c_str(), offset, rc);
                perror("seek file");
                return -EIO;
            }
            if (rc != offset) {
                fprintf(stderr, "%s, seek off 0x%lx != 0x%lx.\n", fname.c_str(), rc, offset);
                return -EIO;
            }
        }

        /* read data from file into memory buffer */
        rc = read(fd, buf, bytes);
        if (rc < 0) {
            fprintf(stderr, "%s, read off 0x%lx + 0x%lx failed %zd.\n", fname.c_str(), offset,
                    bytes, rc);
            perror("read file");
            return -EIO;
        }
        if (rc != bytes) {
            fprintf(stderr, "%s, R off 0x%lx, 0x%lx != 0x%lx.\n", fname.c_str(), count, rc, bytes);
            return -EIO;
        }

        count += bytes;
        buf += bytes;
        offset += bytes;
    } while (count < size);

    if (count != size) {
        fprintf(stderr, "%s, R failed 0x%lx != 0x%lx.\n", fname.c_str(), count, size);
        return -EIO;
    }
    return count;
}

void ValidateCommand::print_results(const std::string& devname, uint64_t addr, double total_time,
                                    double avg_time, double size, double result, test_type type,
                                    int verbose) {
    std::string test_type_str = (type == TEST_TYPE_READ) ? "Read" : "Write";
    std::string memory_type = (addr >= ADDR_START_HBM && addr < ADDR_START_DDR) ? "HBM" : "DDR";

    if (verbose) {
        printf(
            "+----------------+----------------+----------------+----------------+----------------+"
            "----------------+-----------------+\n");
        printf(
            "| Test Type      | Device         | Memory Type    | Total Time (ns)| Avg Time (ns)  "
            "| Size (GB)      | Bandwidth (GB/s)|\n");
        printf(
            "+----------------+----------------+----------------+----------------+----------------+"
            "----------------+-----------------+\n");
        printf("| %-14s | %-14s | %-14s | %-14.2f | %-14.2f | %-14.2f | %-14.2f  |\n",
               test_type_str.c_str(), devname.c_str(), memory_type.c_str(), total_time, avg_time,
               size / GB_DIV, result / GB_DIV);
        printf(
            "+----------------+----------------+----------------+----------------+----------------+"
            "----------------+-----------------\n");
    } else {
        printf("+----------------+----------------+-----------------+\n");
        printf("| Test Type      | Memory Type    | Bandwidth (GB/s)|\n");
        printf("+----------------+----------------+-----------------+\n");
        printf("| %-14s | %-14s | %-14.2f  |\n", test_type_str.c_str(), memory_type.c_str(),
               result / GB_DIV);
        printf("+----------------+----------------+-----------------+\n");
    }
}

void ValidateCommand::print_pci_bandwidth(double pci_bw_result) {
    printf("+---------------------------------------------------+\n");
    printf("| Total PCIe Bandwidth (GB/s): %-19.2f |\n", pci_bw_result);
    printf("+---------------------------------------------------+\n");
}

void ValidateCommand::dma_read_thread(thread_data_t* data) {
    test_dma_read(data->devname, data->addr, data->size, data->offset, data->count, data->verbose);
}

void ValidateCommand::dma_write_thread(thread_data_t* data) {
    test_dma_write(data->devname, data->addr, data->size, data->offset, data->count, data->verbose);
}

int ValidateCommand::run_sim_seq_rw_threads(const std::string& devname, uint64_t size,
                                            uint64_t offset, uint64_t count, int verbose) {
    std::cout << "Performing simultaneous RW test for HBM\n";
    thread_data_t args_hbm, args_ddr;
    args_hbm.devname = devname;
    args_hbm.addr = ADDR_START_HBM;
    args_hbm.size = size;
    args_hbm.offset = offset;
    args_hbm.count = count;
    args_hbm.verbose = verbose;
    args_ddr.devname = devname;
    args_ddr.addr = ADDR_START_DDR;
    args_ddr.size = size;
    args_ddr.offset = offset;
    args_ddr.count = count;
    args_ddr.verbose = verbose;

    std::thread thread_read_hbm(&ValidateCommand::dma_read_thread, this, &args_hbm);
    std::thread thread_write_hbm(&ValidateCommand::dma_write_thread, this, &args_hbm);
    thread_read_hbm.join();
    thread_write_hbm.join();

    std::cout << "Performing simultaneous RW test for DDR\n";
    std::thread thread_read_ddr(&ValidateCommand::dma_read_thread, this, &args_ddr);
    std::thread thread_write_ddr(&ValidateCommand::dma_write_thread, this, &args_ddr);
    thread_read_ddr.join();
    thread_write_ddr.join();

    return EXIT_SUCCESS;
}

int ValidateCommand::run_sim_rw_per_memory(const std::string& devname, uint64_t addr, uint64_t size,
                                           uint64_t offset, uint64_t count, int verbose,
                                           const std::string& mem_type) {
    std::cout << "Running bandwidth test for HBM memory\n";
    std::cout << "Running simultaneous write test on multiple HBM channels\n";

    thread_data_t args_chan0, args_chan1;
    args_chan0.devname = devname;
    args_chan0.addr = addr;
    args_chan0.size = size;
    args_chan0.offset = offset;
    args_chan0.count = count;
    args_chan0.verbose = verbose;

    std::thread write_chan_0(&ValidateCommand::dma_write_thread, this, &args_chan0);

    args_chan1.devname = devname;
    if (mem_type == "hbm") {
        args_chan1.addr = addr + (uint64_t)2 * SIZE_PER_HBM_CHANNEL;
    } else {
        args_chan1.addr = ADDR_START_DIMM_DDR;
    }
    args_chan1.size = size;
    args_chan1.offset = offset;
    args_chan1.count = count;
    args_chan1.verbose = verbose;

    std::thread write_chan_1(&ValidateCommand::dma_write_thread, this, &args_chan1);

    write_chan_0.join();
    write_chan_1.join();

    std::thread read_chan_0(&ValidateCommand::dma_read_thread, this, &args_chan0);
    std::thread read_chan_1(&ValidateCommand::dma_read_thread, this, &args_chan1);

    read_chan_0.join();
    read_chan_1.join();

    return EXIT_SUCCESS;
}

int ValidateCommand::run_pcie_bw_test(const std::string& devname, uint64_t size, uint64_t offset,
                                      uint64_t count, int verbose) {
    int ret = EXIT_SUCCESS;
    int num_of_threads = 8;
    std::vector<std::thread> write_threads;
    std::vector<thread_data_t> args_chan(num_of_threads);

    for (int i = 0; i < num_of_threads; i++) {
        args_chan[i].devname = devname;
        args_chan[i].addr = ADDR_START_DDR + i * 0x80000000;  // ADDR_START_HBM
        args_chan[i].size = size;
        args_chan[i].offset = offset;
        args_chan[i].count = count;
        args_chan[i].verbose = verbose;
    }

    for (int i = 0; i < num_of_threads; i++) {
        write_threads.emplace_back(&ValidateCommand::dma_write_thread_pcie, this, &args_chan[i]);
    }

    for (auto& thread : write_threads) {
        if (thread.joinable()) {
            thread.join();
        } else {
            std::cerr << "Error joining write thread" << std::endl;
            ret = EXIT_FAILURE;
        }
    }

    print_pci_bandwidth(pci_bw_result);
    return ret;
}

double ValidateCommand::test_dma_write_pcie(const std::string& devname, uint64_t addr,
                                            uint64_t size, uint64_t offset, uint64_t count,
                                            int verbose) {
    uint64_t i;
    char* buffer = NULL;
    char* allocated = NULL;
    struct timespec ts_start, ts_end;
    int fpga_fd = open(devname.c_str(), O_RDWR);
    double total_time = 0;
    double result;
    double avg_time = 0;
    int ret = EXIT_FAILURE;
    if (fpga_fd < 0) {
        fprintf(stderr, "unable to open device %s, %d.\n", devname.c_str(), fpga_fd);
        return EXIT_FAILURE;
    }
    posix_memalign((void**)&allocated, 4096 /*alignment */, size + 4096);
    if (!allocated) {
        fprintf(stderr, "OOM %lu.\n", size + 4096);
        close(fpga_fd);
        free(allocated);
        return EXIT_FAILURE;
    }
    buffer = allocated + offset;
    if (verbose) {
        fprintf(stdout, "host buffer 0x%lx = %p\n", size + 4096, buffer);
    }
    for (i = 0; i < count; i++) {
        clock_gettime(CLOCK_MONOTONIC, &ts_start);
        ret = write(fpga_fd, buffer, size);
        if (ret < 0) {
            fprintf(stderr, "Could not write to device buffer.\n");
            close(fpga_fd);
            free(allocated);
            return EXIT_FAILURE;
        }
        clock_gettime(CLOCK_MONOTONIC, &ts_end);
        timespec_sub(&ts_end, &ts_start);
        total_time += (ts_end.tv_sec + ((double)ts_end.tv_nsec / 1000000000.0));
        if (verbose) {
            fprintf(stdout, "CLOCK_MONOTONIC %ld.%09ld sec. write %lu bytes\n", ts_end.tv_sec,
                    ts_end.tv_nsec, size);
        }
    }
    avg_time = (double)total_time / (double)count;
    result = ((double)size) / avg_time;
    close(fpga_fd);
    free(allocated);
    return result;
}

void ValidateCommand::dma_write_thread_pcie(thread_data_t* data) {
    double result = test_dma_write_pcie(data->devname, data->addr, data->size, data->offset,
                                        data->count, data->verbose);
    pci_bw_result.fetch_add(result / GB_DIV, std::memory_order_relaxed);
}

void ValidateCommand::timespec_sub(struct timespec* t1, struct timespec* t2) {
    if (timespec_check(t1) < 0) {
        fprintf(stderr, "invalid time #1: %lld.%.9ld.\n", (long long)t1->tv_sec, t1->tv_nsec);
        return;
    }
    if (timespec_check(t2) < 0) {
        fprintf(stderr, "invalid time #2: %lld.%.9ld.\n", (long long)t2->tv_sec, t2->tv_nsec);
        return;
    }
    t1->tv_sec -= t2->tv_sec;
    t1->tv_nsec -= t2->tv_nsec;
    if (t1->tv_nsec >= 1000000000) {
        t1->tv_sec++;
        t1->tv_nsec -= 1000000000;
    } else if (t1->tv_nsec < 0) {
        t1->tv_sec--;
        t1->tv_nsec += 1000000000;
    }
}

int ValidateCommand::timespec_check(struct timespec* t) {
    if ((t->tv_nsec < 0) || (t->tv_nsec >= 1000000000)) return -1;
    return 0;
}