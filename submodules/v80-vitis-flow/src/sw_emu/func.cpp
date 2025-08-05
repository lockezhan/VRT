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

#include "func.hpp"

Func::Func(const std::string& fn_name, std::vector<Arg> args) : name(fn_name), args(args) {}

std::vector<Arg> Func::getArgs() { return args; }

std::string Func::getFunctionPrototype() {
    std::stringstream ss;

    ss << "extern void " << name << "(";
    for (std::size_t i = 0; i < args.size() - 1; i++) {
        ss << args.at(i).getArgType() << " " << args.at(i).getName() << ", ";
    }
    ss << args.at(args.size() - 1).getArgType() << " " << args.at(args.size() - 1).getName()
       << ");\n";

    return ss.str();
}

std::string Func::getName() { return name; }

FunctionCall::FunctionCall(Func fn, std::string fn_name) : function(fn), functionName(fn_name) {}

std::string FunctionCall::getFunctionName() { return functionName; }

Func FunctionCall::getFunction() { return function; }