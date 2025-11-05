#!/usr/bin/env vivado -mode batch -source
# Vivado script to create YOLOv3 PYNQ-Z2 DPU project

# Set project parameters
set proj_name "pynq_z2_dpu"
set proj_dir "."
set board_part "www.digilentinc.com:pynq-z2:part0:1.0"

# Create project
create_project $proj_name $proj_dir -part $board_part -force

# Set board properties
set_property board_part $board_part [current_project]

# Create block design
create_bd_design "system"

# Add Zynq7 Processing System
startgroup
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0
endgroup

# Configure Zynq PS for PYNQ-Z2
startgroup
set_property -dict [list \
    CONFIG.PCW_IMPORT_BOARD_PRESET {} \
    CONFIG.PCW_PRESET_BOARD_PART {numato.com:mimas_a7:part0:1.0} \
    CONFIG.PCW_UIPARAM_DDR_PARTNO {MT41J128M16HA-15E} \
    CONFIG.PCW_UIPARAM_DDR_BUS_WIDTH {16 Bit} \
    CONFIG.PCW_UIPARAM_DDR_USE_INTERNAL_VREF {0} \
    CONFIG.PCW_TTC0_PERIPHERAL_ENABLE {1} \
    CONFIG.PCW_TTC0_TTC0_CLK {CPU_1X} \
    CONFIG.PCW_TTC0_TTC1_CLK {CPU_1X} \
    CONFIG.PCW_TTC0_TTC2_CLK {CPU_1X} \
    CONFIG.PCW_EN_CLK1_PORT {1} \
    CONFIG.PCW_EN_RST1_PORT {1} \
    CONFIG.PCW_EN_CLK2_PORT {0} \
    CONFIG.PCW_EN_RST2_PORT {0} \
    CONFIG.PCW_EN_CLK3_PORT {0} \
    CONFIG.PCW_EN_RST3_PORT {0} \
    CONFIG.PCW_FPGA_FCLK0_ENABLE {1} \
    CONFIG.PCW_FPGA_FCLK1_ENABLE {1} \
    CONFIG.PCW_TTC0_PERIPHERAL_ENABLE {1} \
    CONFIG.PCW_USB0_PERIPHERAL_ENABLE {1} \
    CONFIG.PCW_USB0_USB0_IO {MIO 28..39} \
    CONFIG.PCW_SD0_PERIPHERAL_ENABLE {1} \
    CONFIG.PCW_SD0_SD0_IO {MIO 40..45} \
    CONFIG.PCW_UART1_PERIPHERAL_ENABLE {1} \
    CONFIG.PCW_UART1_UART1_IO {MIO 48..49} \
    CONFIG.PCW_I2C0_PERIPHERAL_ENABLE {1} \
    CONFIG.PCW_I2C0_I2C0_IO {MIO 50..51} \
    CONFIG.PCW_GPIO_MIO_GPIO_ENABLE {1} \
    CONFIG.PCW_GPIO_MIO_GPIO_IO {MIO} \
    CONFIG.PCW_ENET0_PERIPHERAL_ENABLE {1} \
    CONFIG.PCW_ENET0_ENET0_IO {MIO 16..27} \
    CONFIG.PCW_ENET0_GRP_MDIO_ENABLE {1} \
    CONFIG.PCW_ENET0_GRP_MDIO_IO {MIO 52..53} \
    CONFIG.PWD_BOOT_MODE {JTAG} \
    CONFIG.PCW_ACT_FPGA0_PERIPHERAL_CLKSRC {IO PLL} \
    CONFIG.PCW_ACT_FPGA1_PERIPHERAL_CLKSRC {IO PLL} \
    CONFIG.PCW_ACT_FPGA0_PERIPHERAL_FREQMHZ {100} \
    CONFIG.PCW_ACT_FPGA1_PERIPHERAL_FREQMHZ {150} \
    CONFIG.PCW_USE_DEFAULT_ACLK_PERIPHERAL_FREQMHZ {100} \
    CONFIG.PCW_USE_FABRIC_INTERRUPT {1} \
    CONFIG.PCW_IRQ_F2P_INTR {1} \
] [get_bd_cells processing_system7_0]
endgroup

# Apply board preset
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 -config {
    make_external {FIXED_IO, DDR}
    Master Disable
    Slave Disable
} [get_bd_cells processing_system7_0]

# Add DPU IP (assuming DPU IP is installed)
# Note: This requires DPU IP from Xilinx Vitis AI
startgroup
create_bd_cell -type ip -vlnv xilinx.com:ip:dpucvdx:1.0 dpucvdx_0
endgroup

# Configure DPU
set_property -dict [list \
    CONFIG.DPU_ARCH_SUFFIX {B1152} \
    CONFIG.DPU_CHANNEL_NUM {8} \
    CONFIG.DPU_INPUT_WIDTH {16} \
    CONFIG.DPU_INPUT_HEIGHT {16} \
    CONFIG.DPU_POOLING_EN {true} \
    CONFIG.DPU_RELU_EN {true} \
    CONFIG.DPU_SATURATION_EN {true} \
] [get_bd_cells dpucvdx_0]

# Add AXI Interconnect for DPU
startgroup
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_interconnect_0
endgroup

set_property -dict [list \
    CONFIG.NUM_MI {1} \
] [get_bd_cells axi_interconnect_0]

# Add Processing System Reset
startgroup
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 proc_sys_reset_0
endgroup

# Add Clock Wizard for DPU clocks
startgroup
create_bd_cell -type ip -vlnv xilinx.com:ip:clk_wiz:6.0 clk_wiz_0
endgroup

set_property -dict [list \
    CONFIG.PRIM_SOURCE {No_buffer} \
    CONFIG.CLKOUT1_REQUESTED_OUT_FREQ {150.0} \
    CONFIG.CLKOUT2_REQUESTED_OUT_FREQ {300.0} \
    CONFIG.CLKOUT2_USED {true} \
    CONFIG.USE_LOCKED {false} \
    CONFIG.USE_RESET {false} \
] [get_bd_cells clk_wiz_0]

# Connect clocks
connect_bd_net -net processing_system7_0_FCLK_CLK0 [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] [get_bd_pins axi_interconnect_0/ACLK] [get_bd_pins axi_interconnect_0/S00_ACLK] [get_bd_pins axi_interconnect_0/M00_ACLK] [get_bd_pins proc_sys_reset_0/slowest_sync_clk]

connect_bd_net -net processing_system7_0_FCLK_CLK1 [get_bd_pins processing_system7_0/FCLK_CLK1] [get_bd_pins clk_wiz_0/clk_in1]

connect_bd_net -net clk_wiz_0_clk_out1 [get_bd_pins clk_wiz_0/clk_out1] [get_bd_pins dpucvdx_0/dpu_clk]

connect_bd_net -net clk_wiz_0_clk_out2 [get_bd_pins clk_wiz_0/clk_out2] [get_bd_pins dpucvdx_0/dpu_dsp_clk]

# Connect resets
connect_bd_net -net processing_system7_0_FCLK_RESET0_N [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins proc_sys_reset_0/ext_reset_in]

connect_bd_net -net proc_sys_reset_0_peripheral_aresetn [get_bd_pins proc_sys_reset_0/peripheral_aresetn] [get_bd_pins axi_interconnect_0/ARESETN] [get_bd_pins axi_interconnect_0/S00_ARESETN] [get_bd_pins axi_interconnect_0/M00_ARESETN]

# Connect AXI interfaces
connect_bd_intf_net -net processing_system7_0_M_AXI_GP0 [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins axi_interconnect_0/S00_AXI]

connect_bd_intf_net -net axi_interconnect_0_M00_AXI [get_bd_intf_pins axi_interconnect_0/M00_AXI] [get_bd_intf_pins dpucvdx_0/s_axi]

# Connect DPU interrupt
connect_bd_net -net dpucvdx_0_interrupt [get_bd_pins dpucvdx_0/interrupt] [get_bd_pins processing_system7_0/IRQ_F2P]

# Assign addresses
assign_bd_address

# Create wrapper
make_wrapper -files [get_files $proj_dir/$proj_name.srcs/sources_1/bd/system/system.bd] -top

# Add wrapper to project
add_files -norecurse $proj_dir/$proj_name.srcs/sources_1/bd/system/hdl/system_wrapper.v

# Set top module
set_property top system_wrapper [current_fileset]

# Update compile order
update_compile_order -fileset sources_1

# Save block design
save_bd_design

# Validate design
validate_bd_design

puts "Vivado project created successfully!"
puts "Project: $proj_name"
puts "Board: PYNQ-Z2"
puts "DPU: B1152 configuration"
puts ""
puts "Next steps:"
puts "1. Open project in Vivado GUI: vivado $proj_name.xpr"
puts "2. Generate bitstream"
puts "3. Export hardware (.hdf file)"
puts "4. Use .hdf for PetaLinux build"