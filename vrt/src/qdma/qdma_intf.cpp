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

#include "qdma/qdma_intf.hpp"

namespace vrt {

QdmaIntf::QdmaIntf(const std::string& bdf) {
    this->queueIdx = 0;
    this->bdf = bdf;
    char* bus = strip(bdf.c_str());

    char formattedQueueName[256];
    sprintf(formattedQueueName, QDMA_DEFAULT_QUEUE, bus);
    queueName = std::string(formattedQueueName);
    free(bus);
}

QdmaIntf::QdmaIntf(const std::string& bdf, const uint32_t queueIdx) {
    this->bdf = bdf;
    char* bus = strip(bdf.c_str());

    char formattedQueueName[256];
    sprintf(formattedQueueName, QDMA_DEFAULT_ST_QUEUE, bus, queueIdx);
    queueName = std::string(formattedQueueName);
    free(bus);

    this->queueIdx = queueIdx;
}

QdmaIntf::~QdmaIntf() {}

char* QdmaIntf::strip(const char* bdf) {
    char* output = (char*)malloc(3 * sizeof(char));
    if (sscanf(bdf, "%2[^:]", output) == 1) {
        output[2] = '\0';
    }
    return output;
}

char* QdmaIntf::create_qdma_queue(const char* bdf) {
    char* id = strip(bdf);
    char qdma_queue_name[12];
    sprintf(qdma_queue_name, QDMA_QUEUE_NAME, id);
    char cmd_add[256];
    sprintf(cmd_add, "dma-ctl %s q add idx 0 mode mm dir bi > /dev/null", qdma_queue_name);
    system(cmd_add);
    usleep(1000);
    char cmd_start[256];
    sprintf(cmd_start, "dma-ctl %s q start idx 0 idx_ringsz 15 dir bi > /dev/null",
            qdma_queue_name);
    system(cmd_start);
    usleep(1000);
    char* output = (char*)malloc(256 * sizeof(char));
    sprintf(output, QDMA_DEFAULT_QUEUE, id);
    free(id);
    return output;
}

int QdmaIntf::delete_qdma_queue(const char* bdf) {
    usleep(100000);
    char* id = strip(bdf);
    char cmd_stop[256];
    char qdma_queue_name[12];
    sprintf(qdma_queue_name, QDMA_QUEUE_NAME, id);
    sprintf(cmd_stop, "dma-ctl %s q stop idx 0 dir bi > /dev/null", qdma_queue_name);
    system(cmd_stop);
    usleep(1000);
    char cmd_del[256];
    sprintf(cmd_del, "dma-ctl %s q del idx 0 dir bi > /dev/null", qdma_queue_name);
    system(cmd_del);
    usleep(1000);
    free(id);
    return EXIT_SUCCESS;
}

ssize_t QdmaIntf::write_from_buffer(const char* fname, char* buffer, uint64_t size, uint64_t base) {
    int fd = open(queueName.c_str(), O_WRONLY);
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
                utils::Logger::log(utils::LogLevel::ERROR, __PRETTY_FUNCTION__,
                                   "Could not write to {}", fname);
                return -EIO;
            }
            if (rc != offset) {
                utils::Logger::log(utils::LogLevel::ERROR, __PRETTY_FUNCTION__,
                                   "Could not write to {}", fname);
                return -EIO;
            }
        }

        /* write data to file from memory buffer */
        rc = write(fd, buf, bytes);
        if (rc < 0) {
            utils::Logger::log(utils::LogLevel::ERROR, __PRETTY_FUNCTION__, "Could not write to {}",
                               fname);
            return -EIO;
        }
        if (rc != bytes) {
            utils::Logger::log(utils::LogLevel::ERROR, __PRETTY_FUNCTION__, "Could not write to {}",
                               fname);
            return -EIO;
        }

        count += bytes;
        buf += bytes;
        offset += bytes;
    } while (count < size);

    if (count != size) {
        utils::Logger::log(utils::LogLevel::ERROR, __PRETTY_FUNCTION__, "Could not write to {}",
                           fname);
        return -EIO;
    }
    close(fd);
    return count;
}

ssize_t QdmaIntf::read_to_buffer(const char* fname, char* buffer, uint64_t size, uint64_t base) {
    int fd = open(queueName.c_str(), O_RDONLY);
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
                utils::Logger::log(utils::LogLevel::ERROR, __PRETTY_FUNCTION__,
                                   "Could not read from {}", fname);
                return -EIO;
            }
            if (rc != offset) {
                utils::Logger::log(utils::LogLevel::ERROR, __PRETTY_FUNCTION__,
                                   "Could not read from {}", fname);
                return -EIO;
            }
        }

        /* read data from file into memory buffer */
        rc = read(fd, buf, bytes);
        if (rc < 0) {
            utils::Logger::log(utils::LogLevel::ERROR, __PRETTY_FUNCTION__,
                               "Could not read from {}", fname);
            return -EIO;
        }
        if (rc != bytes) {
            utils::Logger::log(utils::LogLevel::ERROR, __PRETTY_FUNCTION__,
                               "Could not read from {}", fname);
            return -EIO;
        }

        count += bytes;
        buf += bytes;
        offset += bytes;
    } while (count < size);

    if (count != size) {
        utils::Logger::log(utils::LogLevel::ERROR, __PRETTY_FUNCTION__, "Could not read from {}",
                           fname);
        return -EIO;
    }
    close(fd);
    return count;
}

void QdmaIntf::write_buff(char* buffer, uint64_t start_addr, uint64_t size) {
    utils::Logger::log(utils::LogLevel::DEBUG, __PRETTY_FUNCTION__,
                       "Writing buffer with size: {x} to {} at address {x}", size, queueName,
                       start_addr);
    write_from_buffer(queueName.c_str(), buffer, size, start_addr);
}

void QdmaIntf::read_buff(char* buffer, uint64_t start_addr, uint64_t size) {
    utils::Logger::log(utils::LogLevel::DEBUG, __PRETTY_FUNCTION__,
                       "Reading buffer with size: {x} to {} at address {x}", size, queueName,
                       start_addr);
    read_to_buffer(queueName.c_str(), buffer, size, start_addr);
}

uint32_t QdmaIntf::getQueueIdx() { return queueIdx; }

}  // namespace vrt