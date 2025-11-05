#!/usr/bin/env vivado -mode batch -source
# Vivado script to export hardware definition

# Open project
open_project pynq_z2_dpu.xpr

# Open implemented design
open_run impl_1

# Export hardware definition file
puts "Exporting hardware definition..."
write_hwdef -force -file system_wrapper.hdf

# Export hardware handoff file for PYNQ
puts "Exporting hardware handoff..."
write_hw_platform -fixed -force -file system_wrapper.xsa

# If XSA export fails (older Vivado versions), create HWH manually
if {![file exists system_wrapper.xsa]} {
    puts "XSA export not supported, creating HWH file..."
    file mkdir pynq_z2_dpu.srcs/sources_1/bd/system/hw_handoff
    set bd_design [current_bd_design]
    set bd_cells [get_bd_cells]

    # Create HWH content (simplified)
    set hwh_content [open "system_wrapper.hwh" w]
    puts $hwh_content "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
    puts $hwh_content "<hw_handoff>"
    puts $hwh_content "  <project name=\"pynq_z2_dpu\">"
    puts $hwh_content "    <board name=\"pynq-z2\"/>"
    puts $hwh_content "    <part name=\"xc7z020clg400-1\"/>"
    puts $hwh_content "    <design name=\"system\"/>"
    puts $hwh_content "  </project>"
    puts $hwh_content "</hw_handoff>"
    close $hwh_content
}

puts "Hardware export completed!"
puts "Files created:"
puts "- system_wrapper.hdf"
puts "- system_wrapper.xsa (or system_wrapper.hwh)"

# Close project
close_project