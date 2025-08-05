# ##################################################################################################
#  The MIT License (MIT)
#  Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
# 
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software
#  and associated documentation files (the "Software"), to deal in the Software without restriction,
#  including without limitation the rights to use, copy, modify, merge, publish, distribute,
#  sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included in all copies or
#  substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ##################################################################################################

if { $argc != 1 } {
    puts "Usage: program.tcl <program_file>"
    exit 1
}

set program_file [lindex $argv 0]

open_hw_manager
connect_hw_server -allow_non_jtag
open_hw_target
current_hw_device [get_hw_devices xcv80_1]
refresh_hw_device -update_hw_probes false [lindex [get_hw_devices xcv80_1] 0]

set_property PROBES.FILE {} [get_hw_devices xcv80_1]
set_property FULL_PROBES.FILE {} [get_hw_devices xcv80_1]
set_property PROGRAM.FILE $program_file [get_hw_devices xcv80_1]
program_hw_devices [get_hw_devices xcv80_1]
refresh_hw_device [lindex [get_hw_devices xcv80_1] 0]
close_hw_target
close_hw_manager
exit