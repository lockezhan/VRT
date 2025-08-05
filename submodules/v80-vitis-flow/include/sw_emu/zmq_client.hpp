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

#ifndef ZMQ_CLIENT_HPP
#define ZMQ_CLIENT_HPP

#include <string>
#include <zmq.hpp>

/**
 * @brief Client class for ZeroMQ messaging.
 *
 * This class provides a client implementation for ZeroMQ-based communication,
 * allowing software components to communicate with hardware emulators or simulators.
 * It handles connection setup, message reception, and acknowledgments.
 */
class ZmqClient {
   private:
    zmq::context_t context;  ///< ZeroMQ context for socket management
    zmq::socket_t socket;    ///< ZeroMQ socket for communication

   public:
    /**
     * @brief Default constructor for ZmqClient.
     *
     * Initializes the ZeroMQ context and creates a REQ-type socket
     * for request-reply communication pattern.
     */
    ZmqClient();

    /**
     * @brief Destructor for ZmqClient.
     *
     * Closes the socket and terminates the ZeroMQ context.
     */
    ~ZmqClient();

    /**
     * @brief Connects to a ZeroMQ server.
     * @param address String containing the server address (e.g., "tcp://localhost:5555").
     *
     * This method establishes a connection to a ZeroMQ server at the specified address.
     */
    void connect(const std::string& address);

    /**
     * @brief Receives a message from the server.
     * @return String containing the received message.
     *
     * This method blocks until a message is received from the connected server.
     */
    std::string recv();

    /**
     * @brief Sends an acknowledgment message to the server.
     * @param ackMessage String containing the acknowledgment message.
     *
     * This method sends an acknowledgment message back to the server
     * to complete the request-reply communication pattern.
     */
    void ack(const std::string& ackMessage);
};

#endif  // ZMQ_CLIENT_HPP