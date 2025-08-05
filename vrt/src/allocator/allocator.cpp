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

#include "allocator/allocator.hpp"

namespace vrt {
Superblock::Superblock(uint64_t startAddress, uint64_t size)
    : startAddress(startAddress), size(size), offset(0) {}

uint64_t Superblock::allocate(uint64_t size) {
    if (!freeList.empty()) {
        uint64_t addr = freeList.back();
        freeList.pop_back();
        return addr;
    }
    if (offset + size > this->size) {
        throw std::bad_alloc();
    }
    uint64_t addr = startAddress + offset;
    offset += size;
    return addr;
}

void Superblock::deallocate(uint64_t addr) {}

MemoryRange::MemoryRange(uint64_t startAddress, uint64_t size)
    : startAddress(startAddress), size(size), offset(0) {}

Allocator::Allocator(uint64_t superblockSize) : superblockSize(superblockSize) {
    addMemoryRange(MemoryRangeType::HBM, HBM_START, HBM_SIZE);
    addMemoryRange(MemoryRangeType::DDR, DDR_START, DDR_SIZE);
}

void Allocator::addMemoryRange(MemoryRangeType type, uint64_t startAddress, uint64_t size) {
    memoryRanges.emplace(type, MemoryRange(startAddress, size));
}

uint64_t Allocator::allocate(uint64_t size, MemoryRangeType type) {
    if (type == MemoryRangeType::HBM) {
        return allocate(size, type, 0);
    }
    auto it = memoryRanges.find(type);
    if (it == memoryRanges.end()) {
        throw std::out_of_range("Invalid memory range type");
    }

    MemoryRange& range = it->second;

    if (size < superblockSize / 2) {
        for (auto& superblock : range.superblocks) {
            try {
                uint64_t addr = superblock.allocate(size);
                addrToSuperblock[addr] = &superblock;
                return addr;
            } catch (const std::bad_alloc&) {
                continue;
            }
        }
        if (range.offset + superblockSize > range.size) {
            throw std::bad_alloc();
        }
        range.superblocks.emplace_back(range.startAddress + range.offset, superblockSize);
        range.offset += superblockSize;
        uint64_t addr = range.superblocks.back().allocate(size);
        addrToSuperblock[addr] = &range.superblocks.back();
        return addr;
    } else {
        if (!range.freeList.empty()) {
            uint64_t addr = range.freeList.back();
            range.freeList.pop_back();
            return addr;
        }
        if (range.offset + size > range.size) {
            throw std::bad_alloc();
        }
        uint64_t addr = range.startAddress;
        while (addr + size <= range.startAddress + range.size) {
            bool isOccupied = std::any_of(
                range.usedMemoryBlocks.begin(), range.usedMemoryBlocks.end(),
                [addr, size](const std::pair<uint64_t, uint64_t>& block) {
                    return (addr >= block.first && addr < block.first + block.second) ||
                           (addr + size > block.first && addr + size <= block.first + block.second);
                });

            if (!isOccupied) {
                range.usedMemoryBlocks.push_back({addr, size});
                return addr;
            }

            addr += size;
        }

        throw std::bad_alloc();
    }
}

void Allocator::deallocate(uint64_t addr) {
    auto it = addrToSuperblock.find(addr);
    if (it != addrToSuperblock.end()) {
        it->second->deallocate(addr);
        addrToSuperblock.erase(it);
    } else {
        for (auto& [type, range] : memoryRanges) {
            if (addr >= range.startAddress && addr < range.startAddress + range.size) {
                range.freeList.push_back(addr);
                return;
            }
        }
    }
}

uint64_t Allocator::allocate(uint64_t size, MemoryRangeType type, uint8_t port) {
    auto it = memoryRanges.find(type);
    if (it == memoryRanges.end()) {
        throw std::out_of_range("Invalid memory range type");
    }

    if (port > 31) {
        throw std::out_of_range("Invalid port number");
    }

    if (type != MemoryRangeType::HBM) {
        return allocate(size, type);
    }

    MemoryRange& range = it->second;
    uint64_t portBaseAddress = HBM_START + port * HBM_PORT_SIZE;
    uint64_t portEndAddress =
        portBaseAddress + 2 * HBM_PORT_SIZE * 8;  // allow moving to next port (aligned at 64bit)

    if (size < superblockSize / 2) {
        for (auto& superblock : range.superblocks) {
            if (superblock.startAddress < portBaseAddress ||
                superblock.startAddress >= portBaseAddress + HBM_PORT_SIZE) {
                continue;
            }
            try {
                uint64_t addr = superblock.allocate(size);
                addrToSuperblock[addr] = &superblock;
                return addr;
            } catch (const std::bad_alloc&) {
                continue;
            }
        }
        if (range.offset + superblockSize > range.size) {
            throw std::bad_alloc();
        }
        uint64_t sbs = superblockSize;
        // Check if the range is already occupied in usedMemoryBlocks
        bool isOccupied =
            std::any_of(range.usedMemoryBlocks.begin(), range.usedMemoryBlocks.end(),
                        [portBaseAddress, sbs](const std::pair<uint64_t, uint64_t>& block) {
                            return (portBaseAddress >= block.first &&
                                    portBaseAddress < block.first + block.second) ||
                                   (portBaseAddress + sbs > block.first &&
                                    portBaseAddress + sbs <= block.first + block.second);
                        });

        if (!isOccupied) {
            // Allocate a new superblock from portBaseAddress
            range.superblocks.emplace_back(portBaseAddress, superblockSize);
            uint64_t addr = range.superblocks.back().allocate(size);
            addrToSuperblock[addr] = &range.superblocks.back();
            range.usedMemoryBlocks.push_back({addr, superblockSize});
            return addr;
        } else {
            // Find the next free block
            uint64_t nextFreeAddress = portBaseAddress;
            for (const auto& block : range.usedMemoryBlocks) {
                if (nextFreeAddress >= block.first &&
                    nextFreeAddress < block.first + block.second) {
                    nextFreeAddress = block.first + block.second;
                }
            }

            // Check if the next free block is within the range
            if (nextFreeAddress + superblockSize <= range.startAddress + range.size) {
                range.superblocks.emplace_back(nextFreeAddress, superblockSize);
                uint64_t addr = range.superblocks.back().allocate(size);
                addrToSuperblock[addr] = &range.superblocks.back();
                range.usedMemoryBlocks.push_back({addr, size});
                return addr;
            } else {
                throw std::bad_alloc();
            }
        }

    } else {
        if (!range.freeList.empty()) {
            auto it =
                std::find_if(range.freeList.begin(), range.freeList.end(),
                             [portBaseAddress](uint64_t addr) { return addr > portBaseAddress; });

            if (it != range.freeList.end()) {
                uint64_t foundAddress = *it;
                range.freeList.erase(it);  // Remove the allocated address from the free list
                return foundAddress;
            }
        }
        // Search in usedMemoryBlocks for the next free address within the port range
        uint64_t nextFreeAddress = portBaseAddress;
        for (const auto& block : range.usedMemoryBlocks) {
            if (block.first + block.second >= portBaseAddress &&
                block.first + block.second < portEndAddress) {
                nextFreeAddress = block.first + block.second;
            }
        }

        // Check if the next free block is within the port range
        if (nextFreeAddress + size <= portEndAddress) {
            range.usedMemoryBlocks.push_back(
                {nextFreeAddress, size});  // Keep track of the used block
            return nextFreeAddress;
        }

        // Allocate from portBaseAddress
        uint64_t addr = portBaseAddress;
        while (addr + size <= range.startAddress + range.size) {
            // Check if the address range is occupied
            bool isOccupied = std::any_of(
                range.usedMemoryBlocks.begin(), range.usedMemoryBlocks.end(),
                [addr, size](const std::pair<uint64_t, uint64_t>& block) {
                    return (addr >= block.first && addr < block.first + block.second) ||
                           (addr + size > block.first && addr + size <= block.first + block.second);
                });

            if (!isOccupied) {
                range.usedMemoryBlocks.push_back({addr, size});  // Keep track of the used block
                return addr;
            }

            addr += size;  // Move to the next block
        }

        throw std::bad_alloc();
    }
}

uint64_t Allocator::getSize(MemoryRangeType type) const {
    auto it = memoryRanges.find(type);
    if (it == memoryRanges.end()) {
        throw std::out_of_range("Invalid memory range type");
    }
    return it->second.size;
}

}  // namespace vrt
