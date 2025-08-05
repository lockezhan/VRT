# PCIe Hotplug Driver

This project contains a kernel module for PCIe hotplug functionality. The kernel module is located in the `driver` directory.

## Building

To build the kernel module, run:

```
make
```

## Cleaning

To clean up the build artifacts, run:

```
make clean
```

## Installing the Kernel Module

To install the kernel module, run:

```
sudo make install
```
Hotplug driver insertion automatically searches for devices with VENDOR_ID and DEVICE_ID defined in driver/pcie_hotplug_ids.h. To add another device, add your VENDOR_ID and DEVICE_ID above the last line:

```C
    { PCI_DEVICE(0x10ee, 0x50b4) }, /**< Xilinx V80 PCIe device */
    { PCI_DEVICE(0xXXXX, 0xXXXX) }, /**< Add new devices here */
    { 0, } /**< End of list */
```

Then re-run make and sudo make install to reinstall the driver.
## Uninstalling the Kernel Module

To uninstall the kernel module, run:

```
sudo make uninstall
```

## Makefile Targets

- `all`: Builds both the kernel module and the user-space application.
- `clean`: Cleans up the build artifacts.
- `install`: Installs the kernel module to the appropriate directory and updates the module dependencies.
- `uninstall`: Removes the kernel module and cleans up the installation.

## Example Usage
After installing the driver, a device file should be present: `/dev/pcie_hotplug`. You can interact with it as follows:
- `echo 'remove' > /dev/pcie_hotplug_0000:bb:dd.f`. This removes the device from the root tree.
- `echo 'toggle_sbr' > /dev/pcie_hotplug_0000:bb:dd.f`. This triggers PCIe Secondary Bus Reset.
- `echo 'rescan' > /dev/pcie_hotplug_0000:bb:dd.f`. This rescans the PCIe bus.
- `echo 'hotplug' > /dev/pcie_hotplug_0000:bb:dd.f`. This performs a PCIe device hotplug, which reinitializes bound drivers of the device.


## Notes

- Ensure you have the necessary permissions to install and uninstall kernel modules.
- The adding of the hotplug driver should be done after each reboot.
