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

#include "arg.hpp"

#include <regex>

Arg::Arg(std::string name, uint32_t index, std::string argType)
    : name(name), index(index), argType(argType) {
    // Check and convert argType
    std::regex stream_regex(R"(stream<([^,]+), 0>)");
    std::regex stream_ref_regex(R"(stream<([^,]+), 0>&)");

    if (std::regex_match(argType, stream_regex)) {
        this->argType = std::regex_replace(argType, stream_regex, "hls::stream<$1>");
    } else if (std::regex_match(argType, stream_ref_regex)) {
        this->argType = std::regex_replace(argType, stream_ref_regex, "hls::stream<$1>&");
    }
}

std::string Arg::getName() { return name; }

uint32_t Arg::getIndex() { return index; }

std::string Arg::getArgType() { return argType; }