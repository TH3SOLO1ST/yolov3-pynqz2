#!/usr/bin/env vivado -mode batch -source
# Vivado script to build bitstream

# Open project
open_project pynq_z2_dpu.xpr

# Set top module
set_property top system_wrapper [current_fileset]

# Update compile order
update_compile_order -fileset sources_1

# Synthesize design
puts "Starting synthesis..."
launch_runs synth_1 -jobs 4
wait_on_run synth_1

# Check synthesis results
if {[get_property PROGRESS [get_runs synth_1]] != "100%"} {
    puts "Error: Synthesis failed"
    exit 1
}

# Implement design
puts "Starting implementation..."
launch_runs impl_1 -jobs 4
wait_on_run impl_1

# Check implementation results
if {[get_property PROGRESS [get_runs impl_1]] != "100%"} {
    puts "Error: Implementation failed"
    exit 1
}

# Generate bitstream
puts "Generating bitstream..."
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1

# Check bitstream generation
if {[get_property PROGRESS [get_runs impl_1]] != "100%"} {
    puts "Error: Bitstream generation failed"
    exit 1
}

puts "Bitstream generated successfully!"

# Report resource utilization
open_run impl_1
report_utilization -file pynq_z2_dpu_utilization.rpt
report_timing_summary -file pynq_z2_dpu_timing.rpt

puts "Utilization and timing reports generated."

# Close project
close_project