# Examples

This directory contains the example projects for VRT. Each example demonstrates different functionalities and usage patterns of the VRT ecosystem:

| ID | Exemplified Feature | Notes |
|------|-----------|----------------|
| 0 | Linking, AXI-Lite control | |
| 1 | Kernels with AXI-MM interfaces | |
| 2 | Freerunning streaming kernels | Emulation not possible |
| 3 | Controlling multiple V80s | Uses VRTBIN of example 0 |
| 4 | Frequency targets | |
| 5 | Memory performance test | Instantiates current maximum number of kernels |

## How to run the examples

Each example project has its own Makefile that automates the build process.

### Makefile recipes

The `Makefile` in each example include several recipes:

- `all`: Runs the `setup`, `hls`, `hw`, `emu`, `sim` and `app` recipes.
- `setup`: Sets up the build environment.
- `hls`: Builds the HLS kernels specified in the `hls` directory.
- `hw`: Builds the hardware design for the specified project.
- `emu`: Builds the emulation design flow for the specified project.
- `sim`: Builds the simulation design flow for the specified project.
- `app`: Builds the runtime application for the specified project.
- `clean`: Cleans up the build environment.
- `hw_all`: Builds all the necessary recipes for the full hardware flow.
- `emu_all`: Builds all the necessary recipes for the emulation flow.
- `sim_all`: Builds all the necessary recipes for the simulation flow.


### Example usage

```
cd 0x_<example_name>/
make <recipe>_all
```
where recipe can be `hw`, `emu` or `sim`.

In order to run an example, navigate to the build directory and an executable named `0x_<example_name>` will be found.

## How to run

The following environment variable needs to be set prior to running any examples:

```
mkdir -p /home/<user>/.ami
export AMI_HOME="/home/<user>/.ami"
source <VIVADO>
source <VITIS>
```
To make the changes persistent, add the commands to .bashrc. Sourcing the Vivado scripts are needed for the hardware builds, whereas vitis is needed for emulation.

In order to run one of the built examples, one must identify the BDF for the V80 and input it into the code.

```
v80-smi list
--------------------------------------------------------------------
Listing V80 devices 
--------------------------------------------------------------------
V80 device found with BDF: 0000:e2:00.0
--------------------------------------------------------------------
V80 device found with BDF: 0000:21:00.0
--------------------------------------------------------------------
```

To run the example, navigate to the `build` directory, and you will find an executable, called `0x_<project_name>`. The format for running is `0x_<project_name> <BDF> <VRTBIN File>`.
