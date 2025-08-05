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

#include "emulator.hpp"

#include <any>
#include <fstream>
#include <functional>
#include <unordered_map>

Emulator::Emulator(std::vector<std::string> paths, std::vector<Kernel> krnls,
                   std::vector<Connection> conns)
    : kernels(krnls), connections(conns) {
    for (auto path : paths) {
        JsonParser parser(path + JSON_PATH);
        functions.emplace_back(parser.getFunction());
    }

    for (auto krnl : kernels) {
        for (auto fn : functions) {
            if (fn.getName() == krnl.getTopModelName()) {
                functionCalls.emplace_back(FunctionCall(fn, krnl.getName()));
            }
        }
    }
    for (auto fnc : functionCalls) {
        std::cout << fnc.getFunctionName() << std::endl;
    }
}

void Emulator::print() {
    std::ofstream out("tb.cpp");
    out << "#include <iostream>\n";
    out << "#include <ap_fixed.h>\n";
    out << "#include <hls_stream.h>\n";
    out << "#include <ap_int.h>\n";
    out << "#include <zmq.hpp>\n";
    out << "#include <json/json.h>\n";
    out << "#include <cstdint>\n";
    out << "#include <unordered_map>\n";
    out << "#include <map>\n";
    out << "#include <vector>\n";
    out << "#include <cstring>\n";
    out << "\n\n";

    for (auto fn : functions) {
        out << fn.getFunctionPrototype() << '\n';
    }

    out << "\n\n";

    out << "template <typename T>\n";
    out << "void assignValue(T& var, const Json::Value& value) {\n";
    out << "\tif (value.isString()) {\n";
    out << "\t\tstd::istringstream iss(value.asString());\n";
    out << "\t\tiss >> var;\n";
    out << "\t} else if (value.isInt()) {\n";
    out << "\t\tvar = static_cast<T>(value.asInt());\n";
    out << "\t} else if (value.isUInt()) {\n";
    out << "\t\tvar = static_cast<T>(value.asUInt());\n";
    out << "\t} else if (value.isDouble()) {\n";
    out << "\t\tvar = static_cast<T>(value.asDouble());\n";
    out << "\t} else {\n";
    out << "\t\tthrow std::runtime_error(\"Unsupported JSON value type\");\n";
    out << "\t}\n";
    out << "}\n\n";

    out << "template <typename T>\n";
    out << "Json::Value createJsonValue(const T& var) {\n";
    out << "\tJson::Value value;\n";
    out << "\tvalue = *reinterpret_cast<const uint32_t*>(&var);\n";
    out << "\treturn value;\n";
    out << "}\n\n";

    out << "Json::Value createJsonBuffer(const uint8_t* buffer, size_t size) {\n";
    out << "\tJson::Value value(Json::arrayValue);\n";
    out << "\tfor (size_t i = 0; i < size; ++i) {\n";
    out << "\t\tvalue.append(buffer[i]);\n";
    out << "\t}\n";
    out << "\treturn value;\n";
    out << "}\n\n";

    out << "int main() {\n";
    out << "\t// Initialize zmq context and socket\n";
    out << "\tzmq::context_t context(1);\n";
    out << "\tzmq::socket_t socket(context, ZMQ_REP);\n";
    out << "\tsocket.bind(\"tcp://*:5555\");\n\n";

    out << "\tstd::map<std::string, void*> buffers;\n";
    out << "\tstd::map<std::string, size_t> bufferSizes;\n";
    out << "\tstd::map<std::string, void*> streamingBuffers;\n";

    std::map<std::string, void*> streamingBuffers;
    std::map<std::string, std::string> streamTypes;
    std::vector<std::string> refVariables;

    // Declare variables for each function parameter
    for (auto fnc : functionCalls) {
        std::string fncName = fnc.getFunctionName();
        for (auto param : fnc.getFunction().getArgs()) {
            std::string argType = param.getArgType();
            std::string argName = fncName + "_" + param.getName();

            // Remove reference symbol '&' if present
            bool isRef = false;
            if (argType.back() == '&') {
                argType.pop_back();
                isRef = true;
            }

            if (argType.find("hls::stream") != std::string::npos) {
                // Extract the data type from the stream type
                std::string streamDataType = argType.substr(
                    argType.find('<') + 1, argType.rfind('>') - argType.find('<') - 1);
                streamTypes[argName] = streamDataType;
                continue;
            } else if (argType.find('*') != std::string::npos) {
                out << "\t" << argType.substr(0, argType.find('*')) << "* " << argName << ";\n";
            } else {
                out << "\t" << argType << " " << argName << ";\n";
                if (isRef) {
                    refVariables.push_back(argName);
                }
            }
        }
    }

    out << "\n";

    // Declare hls::stream variables for each unique stream type
    std::map<std::string, std::string> declaredStreams;
    int streamCounter = 0, streamingBufferCounter = 0;
    for (const auto& connection : connections) {
        std::string srcStream = connection.src.kernelName + "_" + connection.src.interfaceName;
        std::string dstStream = connection.dst.kernelName + "_" + connection.dst.interfaceName;
        std::string streamDataType = streamTypes[srcStream];
        if (streamDataType == "") {
            streamDataType = streamTypes[dstStream];
        }
        if (streamDataType == "") {  // both connections made to cips which is 512bit by default
            streamDataType = "ap_uint<512>";
        }

        if (declaredStreams.find(srcStream) == declaredStreams.end()) {
            std::string streamName;  // = "stream_" + std::to_string(streamCounter++);
            if (connection.src.kernelName == "cips") {
                std::regex re("qdma_(\\d)");
                std::smatch match;
                if (std::regex_search(connection.src.interfaceName, match, re) &&
                    match.size() > 1) {
                    std::string number = match.str(1);
                    streamName = "streamingBuffer_" + number;
                } else {
                    std::cerr << "Pattern not found in string: " << connection.src.interfaceName
                              << std::endl;
                    continue;
                }
                // streamName = "streamingBuffer_" + std::to_string(streamingBufferCounter++);
                out << "\thls::stream<ap_uint<512>> " << streamName << ";\n";
                declaredStreams[srcStream] = streamName;
                declaredStreams[dstStream] = streamName;
                streamTypes[streamName] = "ap_uint<512>";  // Populate streamTypes map
                streamingBuffers[streamName] = nullptr;    // change this
            } else if (connection.dst.kernelName == "cips") {
                std::regex re("qdma_(\\d)");
                std::smatch match;
                if (std::regex_search(connection.dst.interfaceName, match, re) &&
                    match.size() > 1) {
                    std::string number = match.str(1);
                    streamName = "outputStreamingBuffer_" + number;
                } else {
                    // Handle the case where the pattern is not found
                    std::cerr << "Pattern not found in string: " << connection.dst.interfaceName
                              << std::endl;
                    continue;  // or break, depending on how you want to handle this case
                }
                // streamName = "streamingBuffer_" + std::to_string(streamingBufferCounter++);
                out << "\thls::stream<ap_uint<512>> " << streamName << ";\n";
                declaredStreams[srcStream] = streamName;
                declaredStreams[dstStream] = streamName;
                streamTypes[streamName] = "ap_uint<512>";  // Populate streamTypes map
                streamingBuffers[streamName] = nullptr;    // change this
            } else {
                streamName = "stream_" + std::to_string(streamCounter++);
                out << "\thls::stream<" << streamDataType << "> " << streamName << ";\n";
                declaredStreams[srcStream] = streamName;
                declaredStreams[dstStream] = streamName;
                streamTypes[streamName] = streamDataType;  // Populate streamTypes map

                streamingBuffers[streamName] = nullptr;  // change this
            }
        } else {
            declaredStreams[dstStream] = declaredStreams[srcStream];
        }
    }
    // Create a map to store streaming buffers
    for (const auto& connection : connections) {
        if (connection.src.kernelName == "cips") {
            std::string srcStream = connection.src.kernelName + "_" + connection.src.interfaceName;
            out << "\tstreamingBuffers[\"" << declaredStreams[srcStream]
                << "\"] = reinterpret_cast<void*>(&" << declaredStreams[srcStream] << ");\n";
        } else if (connection.dst.kernelName == "cips") {
            std::string dstStream = connection.dst.kernelName + "_" + connection.dst.interfaceName;
            out << "\tstreamingBuffers[\"" << declaredStreams[dstStream]
                << "\"] = reinterpret_cast<void*>(&" << declaredStreams[dstStream] << ");\n";
        }
    }

    out << "\n\twhile (true) {\n";  // Changed to infinite loop
    out << "\t\tzmq::message_t request;\n";
    out << "\t\tsocket.recv(request);\n";
    out << "\t\tstd::string req_str(static_cast<char*>(request.data()), request.size());\n";
    out << "\t\tJson::Value root;\n";
    out << "\t\tJson::Reader reader;\n";
    out << "\t\treader.parse(req_str, root);\n";
    out << "\t\tstd::string command = root[\"command\"].asString();\n";
    out << "\t\tstd::string argType;\n";
    out << "\t\tif (command == \"populate\") {\n";
    out << "\t\t\tstd::string name = root[\"name\"].asString();\n";
    out << "\t\t\tsize_t bufferSize = root[\"size\"].asUInt64();\n";

    out << "\t\t\tzmq::message_t data;\n";
    out << "\t\t\tsocket.recv(data);\n";
    out << "\t\t\tvoid* buffer = new uint8_t[bufferSize];\n";
    out << "\t\t\tmemcpy(buffer, data.data(), bufferSize);\n";

    out << "\t\t\tbuffers[name] = buffer;\n";
    out << "\t\t\tbufferSizes[name] = bufferSize;\n";
    out << "\t\t\tsocket.send(zmq::message_t(\"OK\", 2), zmq::send_flags::none);\n";  // Send OK
                                                                                      // after
                                                                                      // populate
    out << "\t\t} else if (command == \"stream_in\") {\n";
    out << "\t\t\tstd::string name = root[\"name\"].asString();\n";
    out << "\t\t\tzmq::message_t data;\n";
    out << "\t\t\tsocket.recv(data);\n";
    out << "\t\t\thls::stream<ap_uint<512>>* stream = "
           "reinterpret_cast<hls::stream<ap_uint<512>>*>(streamingBuffers[name]);\n";
    out << "\t\t\tfor (size_t i = 0; i < data.size() / sizeof(ap_uint<512>); i++) {\n";
    out << "\t\t\t\tap_uint<512> value;\n";
    out << "\t\t\t\tmemcpy(&value, static_cast<uint8_t*>(data.data()) + i * sizeof(ap_uint<512>), "
           "sizeof(ap_uint<512>));\n";
    out << "\t\t\t\tstream->write(value);\n";
    out << "\t\t\t}\n";
    out << "\t\t\tsocket.send(zmq::message_t(\"OK\", 2), zmq::send_flags::none);\n";
    out << "\t\t} else if (command == \"stream_out\") {\n";
    out << "\t\t\tstd::string name = root[\"name\"].asString();\n";
    out << "\t\t\tsize_t size = root[\"size\"].asUInt64();\n";
    out << "\t\t\thls::stream<ap_uint<512>>* stream = "
           "reinterpret_cast<hls::stream<ap_uint<512>>*>(streamingBuffers[name]);\n";
    out << "\t\t\tstd::vector<uint8_t> buffer(size);\n";
    out << "\t\t\tfor (size_t i = 0; i < size / sizeof(ap_uint<512>); i++) {\n";
    out << "\t\t\t\tap_uint<512> value = stream->read();\n";
    out << "\t\t\t\tmemcpy(buffer.data() + i * sizeof(ap_uint<512>), &value, "
           "sizeof(ap_uint<512>));\n";
    out << "\t\t\t}\n";
    out << "\t\t\tsocket.send(zmq::message_t(buffer.data(), buffer.size()), "
           "zmq::send_flags::none);\n";
    out << "\t\t} else if (command == \"call\") {\n";
    out << "\t\t\tstd::string functionName = root[\"function\"].asString();\n";

    for (auto fnc : functionCalls) {
        std::string fncName = fnc.getFunctionName();
        out << "\t\t\tif (functionName == \"" << fncName << "\") {\n";
        int idx = 0;
        for (int i = 0; i < fnc.getFunction().getArgs().size(); ++i) {
            std::string argIndex = "arg" + std::to_string(idx);
            std::string argName = fncName + "_" + fnc.getFunction().getArgs()[i].getName();

            if (fnc.getFunction().getArgs()[i].getArgType().find("hls::stream") !=
                std::string::npos) {
                if (streamingBuffers.find(declaredStreams[argName]) != streamingBuffers.end()) {
                    // idx++;
                    // out << "\t\t\t\targType = root[\"args\"][\"" << argIndex <<
                    // "\"][\"type\"].asString();\n";
                }

                continue;
            }

            out << "\t\t\t\targType = root[\"args\"][\"" << argIndex
                << "\"][\"type\"].asString();\n";

            if (fnc.getFunction().getArgs()[i].getArgType().find('*') != std::string::npos) {
                out << "\t\t\t\tif (argType == \"buffer\") {\n";
                out << "\t\t\t\t\tstd::string bufferName = root[\"args\"][\"" << argIndex
                    << "\"][\"name\"].asString();\n";
                out << "\t\t\t\t\tif (buffers.find(bufferName) != buffers.end()) {\n";
                out << "\t\t\t\t\t\t" << argName << " = static_cast<"
                    << fnc.getFunction().getArgs()[i].getArgType().substr(
                           0, fnc.getFunction().getArgs()[i].getArgType().find('*'))
                    << "*>(buffers[bufferName]);\n";
                out << "\t\t\t\t\t}\n";
                out << "\t\t\t\t}\n";
            } else {
                out << "\t\t\t\tif (argType == \"scalar\") {\n";
                out << "\t\t\t\t\tassignValue(" << argName << ", root[\"args\"][\"" << argIndex
                    << "\"][\"value\"]);\n";
                out << "\t\t\t\t}\n";
            }
            idx++;
        }

        out << "\t\t\t\t" << fnc.getFunction().getName() << "(";

        bool firstArg = true;
        for (int i = 0; i < fnc.getFunction().getArgs().size(); ++i) {
            std::string argName = fncName + "_" + fnc.getFunction().getArgs()[i].getName();

            if (!firstArg) {
                out << ", ";
            }
            firstArg = false;

            if (fnc.getFunction().getArgs()[i].getArgType().find("hls::stream") !=
                std::string::npos) {
                out << declaredStreams[argName];
            } else {
                out << argName;
            }
        }

        out << ");\n\t\t\t}\n";
    }

    out << "\t\t\tsocket.send(zmq::message_t(\"OK\", 2), zmq::send_flags::none);\n";  // Send OK
                                                                                      // after call
    out << "\t\t} else if (command == \"fetch\") {\n";
    out << "\t\t\tstd::string type = root[\"type\"].asString();\n";
    out << "\t\t\tJson::Value response;\n";
    out << "\t\t\tif (type == \"scalar\") {\n";
    out << "\t\t\t\tstd::string functionName = root[\"function\"].asString();\n";
    out << "\t\t\t\tstd::string arg = root[\"arg\"].asString();\n";

    for (auto fnc : functionCalls) {
        std::string fncName = fnc.getFunctionName();
        for (int i = 0; i < fnc.getFunction().getArgs().size(); ++i) {
            std::string argIndex = "arg" + std::to_string(i);
            std::string argName = fncName + "_" + fnc.getFunction().getArgs()[i].getName();
            if (fnc.getFunction().getArgs()[i].getArgType().find("hls::stream") ==
                std::string::npos) {
                out << "\t\t\t\tif (functionName == \"" << fncName << "\" && arg == \"" << argIndex
                    << "\") {\n";
                out << "\t\t\t\t\tresponse = createJsonValue(" << argName << ");\n";
                out << "\t\t\t\t}\n";
            }
        }
    }

    out << "\t\t\t} else if (type == \"buffer\") {\n";
    out << "\t\t\t\tstd::string name = root[\"name\"].asString();\n";
    out << "\t\t\t\tif (buffers.find(name) != buffers.end()) {\n";
    out << "\t\t\t\t\tresponse = createJsonBuffer(static_cast<uint8_t*>(buffers[name]), "
           "bufferSizes[name]);\n";
    out << "\t\t\t\t}\n";
    out << "\t\t\t}\n";

    out << "\t\t\tstd::string responseStr = Json::writeString(Json::StreamWriterBuilder(), "
           "response);\n";
    out << "\t\t\tsocket.send(zmq::message_t(responseStr.c_str(), responseStr.size()), "
           "zmq::send_flags::none);\n";
    out << "\t\t} else if (command == \"exit\") {\n";
    out << "\t\t\tsocket.send(zmq::message_t(\"OK\", 2), zmq::send_flags::none);\n";
    out << "\t\t\tbreak;\n";
    out << "\t\t}\n";
    out << "\t}\n";

    // Print reference variables at the end
    for (const auto& var : refVariables) {
        out << "\tstd::cout << \"" << var << ": \" << " << var << " << std::endl;\n";
    }

    out << "\treturn 0;\n";
    out << "}\n";
}