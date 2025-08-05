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

#include "json_parser.hpp"

JsonParser::JsonParser(const std::string& file) {
    std::ifstream jsonFile(file);
    if (!jsonFile.is_open()) {
        utils::Logger::log(utils::LogLevel::ERROR, __PRETTY_FUNCTION__, "Unable to open file {}",
                           file);
        throw std::runtime_error("Unable to open json file");
    }

    Json::Value root;
    jsonFile >> root;

    const Json::Value top = root["Top"];

    if (top.isNull()) {
        utils::Logger::log(utils::LogLevel::ERROR, __PRETTY_FUNCTION__,
                           "Top field not found in json file");
        throw std::runtime_error("Top field not found in json file");
    }
    std::string functionName = top.asString();

    const Json::Value args = root["Args"];
    if (args.isNull()) {
        utils::Logger::log(utils::LogLevel::ERROR, __PRETTY_FUNCTION__,
                           "Args field not found in json file");
        throw std::runtime_error("Args field not found in json file");
    }
    std::vector<JsonFormat> sortedArgs;
    for (const auto& key : args.getMemberNames()) {
        const Json::Value arg = args[key];
        int index = std::stoi(arg["index"].asString());
        sortedArgs.emplace_back(JsonFormat{index, arg, key});
    }
    std::sort(sortedArgs.begin(), sortedArgs.end(), [](auto a, auto b) { return a.idx < b.idx; });

    std::vector<Arg> funcArgs;
    for (const auto& pair : sortedArgs) {
        const Json::Value& arg = pair.value;
        Arg argObj(pair.key, std::stoul(arg["index"].asString()), arg["srcType"].asString());
        funcArgs.push_back(argObj);
        std::string index = arg["index"].asString();
        std::string srcType = arg["srcType"].asString();
    }

    this->func = Func(functionName, funcArgs);
}

Func JsonParser::getFunction() const { return this->func; }