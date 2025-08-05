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

#ifndef FUNC_HPP
#define FUNC_HPP

#include <sstream>
#include <string>
#include <vector>

#include "arg.hpp"

/**
 * @brief Class representing a function for emulation.
 *
 * This class stores information about a function including its name and arguments.
 * It is used in the software emulation environment to represent hardware kernel functions.
 */
class Func {
   private:
    std::vector<Arg> args;  ///< List of arguments for the function
    std::string name;       ///< Name of the function

   public:
    /**
     * @brief Default constructor for Func.
     */
    Func() = default;

    /**
     * @brief Constructor for Func with initialization parameters.
     * @param fn_name Name of the function.
     * @param args Vector of Arg objects representing function arguments.
     */
    Func(const std::string& fn_name, std::vector<Arg> args);

    /**
     * @brief Gets the list of arguments for this function.
     * @return Vector of Arg objects representing function arguments.
     */
    std::vector<Arg> getArgs();

    /**
     * @brief Generates a function prototype for this function.
     * @return String containing the function prototype.
     */
    std::string getFunctionPrototype();

    /**
     * @brief Gets the name of the function.
     * @return String containing the function name.
     */
    std::string getName();
};

/**
 * @brief Class representing a function call in an emulation sequence.
 *
 * This class represents a specific invocation of a function during emulation,
 * storing the function object and its name for execution tracking.
 */
class FunctionCall {
   private:
    Func function;             ///< The function being called
    std::string functionName;  ///< Name of the function being called

   public:
    /**
     * @brief Constructor for FunctionCall with initialization parameters.
     * @param fn Function object representing the function to call.
     * @param fn_name Name of the function being called.
     */
    FunctionCall(Func fn, std::string fn_name);

    /**
     * @brief Gets the name of the function being called.
     * @return String containing the function name.
     */
    std::string getFunctionName();

    /**
     * @brief Gets the function object for this call.
     * @return Func object representing the function being called.
     */
    Func getFunction();
};

#endif  // FUNC_HPP