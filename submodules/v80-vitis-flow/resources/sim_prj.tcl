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

set src_dir        "[pwd]/sim"
set bd_name        "top"
create_project sim_prj "[pwd]/sim/sim_prj/" -part xcv80-lsva4737-2MHP-e-S -force

add_files -norecurse ../resources/sim_mem.v
update_compile_order -fileset sources_1
update_compile_order -fileset sim_1

set_property ip_repo_paths "${src_dir}/iprepo" [current_project]
update_ip_catalog
update_compile_order -fileset sources_1
update_compile_order -fileset sim_1
create_bd_design ${bd_name}
current_bd_design ${bd_name}

create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 s_axi_ctrl
set_property -dict [list CONFIG.ADDR_WIDTH 64] [get_bd_intf_ports s_axi_ctrl]
set_property -dict [list CONFIG.HAS_BURST 0 CONFIG.HAS_CACHE 0 CONFIG.HAS_LOCK 0 CONFIG.HAS_PROT 0 CONFIG.HAS_QOS 0 CONFIG.HAS_REGION 0] [get_bd_intf_ports s_axi_ctrl]

create_bd_port -dir I -type clk clk
create_bd_port -dir I -type rst rst
create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 mem
set_property -dict [list CONFIG.ADDR_WIDTH 64] [get_bd_intf_ports mem]
set_property -dict [list CONFIG.READ_WRITE_MODE READ_WRITE] [get_bd_intf_ports mem]
set_property -dict [list CONFIG.DATA_WIDTH 64] [get_bd_intf_ports mem]

create_bd_cell -type ip -vlnv xilinx.com:ip:axi_bram_ctrl:4.1 bram_ctrl
set_property -dict [list CONFIG.SINGLE_PORT_BRAM {0} CONFIG.DATA_WIDTH {64} CONFIG.ECC_TYPE {0} CONFIG.READ_LATENCY {50}] [get_bd_cells bram_ctrl]

create_bd_cell -type module -reference sim_mem sim_mem_0

connect_bd_intf_net [get_bd_intf_pins bram_ctrl/BRAM_PORTA] [get_bd_intf_pins sim_mem_0/MEM_PORT_A]
connect_bd_intf_net [get_bd_intf_pins bram_ctrl/BRAM_PORTB] [get_bd_intf_pins sim_mem_0/MEM_PORT_B]

set axi_sc [ create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect axi_sc ]
set_property CONFIG.NUM_SI {1} [get_bd_cells axi_sc] 
set mem_sc [ create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect mem_sc ]

connect_bd_intf_net [get_bd_intf_ports s_axi_ctrl] [get_bd_intf_pins axi_sc/S00_AXI]
connect_bd_net [get_bd_ports clk] [get_bd_pins axi_sc/aclk]
connect_bd_net [get_bd_ports rst] [get_bd_pins axi_sc/aresetn]

connect_bd_intf_net [get_bd_intf_pins mem_sc/M00_AXI] [get_bd_intf_pins bram_ctrl/S_AXI]
connect_bd_intf_net [get_bd_intf_pins mem_sc/S00_AXI] [get_bd_intf_ports mem]
connect_bd_net [get_bd_ports clk] [get_bd_pins bram_ctrl/s_axi_aclk]
connect_bd_net [get_bd_ports rst] [get_bd_pins bram_ctrl/s_axi_aresetn]

connect_bd_net [get_bd_ports clk] [get_bd_pins mem_sc/aclk]
connect_bd_net [get_bd_ports rst] [get_bd_pins mem_sc/aresetn]