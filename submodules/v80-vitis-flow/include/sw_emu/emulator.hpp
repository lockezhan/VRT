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

#ifndef EMULATOR_HPP
#define EMULATOR_HPP

#include <json/json.h>

#include <regex>
#include <string>
#include <vector>

#include "arg_parser.hpp"
#include "func.hpp"
#include "json_parser.hpp"
#include "kernel.hpp"
#include "logger.hpp"
#include "zmq_client.hpp"

#define JSON_PATH "/sol1_data.json"

/**
 * @brief Class providing a software emulation environment for hardware kernels.
 *
 * This class manages software emulation of hardware accelerator kernels,
 * allowing functionality testing without actual hardware. It parses kernel
 * function definitions, handles function call sequences, and maintains the
 * connections between various kernels in the design.
 */
class Emulator {
   private:
    std::vector<Func> functions;              ///< List of emulated functions
    std::vector<FunctionCall> functionCalls;  ///< Sequence of function calls to emulate
    std::vector<Kernel> kernels;              ///< Hardware kernels to emulate
    std::vector<Connection> connections;      ///< Connections between kernels

   public:
    /**
     * @brief Constructor for Emulator.
     * @param paths Vector of paths to source files containing function definitions.
     * @param krnls Vector of kernel objects to emulate.
     * @param conns Vector of connection objects defining inter-kernel connections.
     */
    Emulator(std::vector<std::string> paths, std::vector<Kernel> krnls,
             std::vector<Connection> conns);

    /**
     * @brief Prints information about the emulation setup.
     *
     * This method outputs details about the loaded functions, kernels,
     * and connections for debugging or documentation purposes.
     */
    void print();
};

#endif  // EMULATOR_HPP