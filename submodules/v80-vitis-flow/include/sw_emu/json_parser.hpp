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

#ifndef JSON_PARSER_HPP
#define JSON_PARSER_HPP

#include <json/json.h>

#include <algorithm>
#include <fstream>
#include <vector>

#include "arg.hpp"
#include "func.hpp"
#include "logger.hpp"

/**
 * @brief Structure to store JSON format information.
 *
 * This structure holds information about JSON elements including their index,
 * key name, and value for processing structured data.
 */
struct JsonFormat {
    int idx;            ///< Index of the JSON element
    Json::Value value;  ///< Value of the JSON element
    std::string key;    ///< Key name of the JSON element
};

/**
 * @brief Class for parsing JSON configuration files for kernel functions.
 *
 * This class parses JSON files that contain function definitions for hardware
 * kernels, extracting information about function names and argument lists.
 */
class JsonParser {
   private:
    Func func;  ///< Function object extracted from the JSON file

   public:
    /**
     * @brief Constructor for JsonParser.
     * @param file Path to the JSON file to parse.
     *
     * This constructor loads and parses the specified JSON file,
     * extracting function information and storing it in the func member.
     */
    JsonParser(const std::string& file);

    /**
     * @brief Gets the parsed function.
     * @return Func object containing the parsed function information.
     */
    Func getFunction() const;
};

#endif  // JSON_PARSER_HPP