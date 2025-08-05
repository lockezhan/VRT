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

#ifndef ALLOCATOR_HPP
#define ALLOCATOR_HPP

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <vector>
namespace vrt {

/**
 * @brief Enum class representing the type of memory range.
 */
enum class MemoryRangeType {
    HBM,  ///< High Bandwidth Memory
    DDR   ///< Double Data Rate Memory
};

/// Starting address of HBM
constexpr uint64_t HBM_START = 0x4000000000;
/// Size of HBM (32 GB)
constexpr uint64_t HBM_SIZE = 32L * 1024 * 1024 * 1024;  // 32G
/// Size of HBM Port (1 GB)
constexpr uint64_t HBM_PORT_SIZE = 1L * 1024 * 1024 * 1024;  // 1G

/// Starting address of DDR DIMM
constexpr uint64_t DDR_START = 0x60000000000;
/// Size of DDR (32 GB)
constexpr uint64_t DDR_SIZE = 32L * 1024 * 1024 * 1024;  // 32G

/**
 * @brief Class representing a superblock of memory.
 */
class Superblock {
   public:
    /**
     * @brief Constructor for Superblock.
     * @param startAddress The starting address of the superblock.
     * @param size The size of the superblock.
     */
    Superblock(uint64_t startAddress, uint64_t size);

    /**
     * @brief Allocates a block of memory from the superblock.
     * @param size The size of the memory block to allocate.
     * @return The starting address of the allocated memory block.
     */
    uint64_t allocate(uint64_t size);

    /**
     * @brief Deallocates a block of memory.
     * @param addr The starting address of the memory block to deallocate.
     */
    void deallocate(uint64_t addr);

    uint64_t startAddress;  ///< The starting address of the superblock.
   private:
    uint64_t size;                   ///< The size of the superblock.
    uint64_t offset;                 ///< The current offset for allocation.
    std::vector<uint64_t> freeList;  ///< List of free memory blocks.
};

/**
 * @brief Struct representing a range of memory.
 */
struct MemoryRange {
    uint64_t startAddress;                ///< The starting address of the memory range.
    uint64_t size;                        ///< The size of the memory range.
    uint64_t offset;                      ///< The current offset for allocation.
    std::vector<Superblock> superblocks;  ///< List of superblocks in the memory range.
    std::vector<uint64_t> freeList;       ///< List of free memory blocks.
    std::vector<std::pair<uint64_t, uint64_t>> usedMemoryBlocks;  ///< List of used memory blocks.
    /**
     * @brief Constructor for MemoryRange.
     * @param startAddress The starting address of the memory range.
     * @param size The size of the memory range.
     */
    MemoryRange(uint64_t startAddress, uint64_t size);
};

/**
 * @brief Class representing a memory allocator.
 */
class Allocator {
   public:
    /**
     * @brief Constructor for Allocator.
     * @param superblockSize The size of the superblocks to use.
     */
    Allocator(uint64_t superblockSize = 4096);

    Allocator() : Allocator(4096) {}

    /**
     * @brief Adds a memory range to the allocator.
     * @param type The type of memory range (HBM or DDR).
     * @param startAddress The starting address of the memory range.
     * @param size The size of the memory range.
     */
    void addMemoryRange(MemoryRangeType type, uint64_t startAddress, uint64_t size);

    /**
     * @brief Allocates a block of memory.
     * @param size The size of the memory block to allocate.
     * @param type The type of memory range to allocate from (HBM or DDR).
     * @return The starting address of the allocated memory block.
     */
    uint64_t allocate(uint64_t size, MemoryRangeType type);

    /**
     * @brief Deallocates a block of memory.
     * @param addr The starting address of the memory block to deallocate.
     */
    void deallocate(uint64_t addr);

    /**
     * @brief Allocates a block of memory from the specified port.
     * @param size The size of the memory block to allocate.
     * @param type The type of memory range to allocate from (HBM or DDR).
     * @param port The port to allocate from.
     * @return The starting address of the allocated memory block.
     */
    uint64_t allocate(uint64_t size, MemoryRangeType type, uint8_t port);

    /**
     * @brief Gets the size of the specified memory range type.
     * @param type The type of memory range (HBM or DDR).
     * @return The size of the specified memory range type.
     */
    uint64_t getSize(MemoryRangeType type) const;

   private:
    uint64_t superblockSize;  ///< The size of the superblocks.
    std::unordered_map<MemoryRangeType, MemoryRange>
        memoryRanges;  ///< Map of memory ranges by type.
    std::unordered_map<uint64_t, Superblock*>
        addrToSuperblock;  ///< Map of addresses to superblocks.
};

}  // namespace vrt

#endif  // ALLOCATOR_HPP