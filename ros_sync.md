# hostname
* pc: book.local
* ELAR-Systems.local
* robot
> export ROS_HOSTNAME=tegra-ubuntu.local
* desktop
> export ROS_HOSTNAME=localhost
> export ROS_MASTER_URI=http://tegra-ubuntu.local:11311

# ssh to robot
* ssh tinker@10.42.0.129
* ssh tinker@10.42.0.43
# ssh cp
* scp /home/king/.vimrc tinker@10.42.0.129:~/
* scp king@10.106.18.46
# key_twist
* rosrun teleop_twist_keyboard teleop_twist_keyboard.py

SUBSYSTEM=="tty", ENV{ID_SERIAL_SHORT}=="0001",MODE="0666", OWNER="tinker", GROUP="tinker",SYMLINK+="rplidar"

# dl
* ssh lin@10.106.18.118

# king desk
* king@10.106.26.139

# king book
* king@10.106.26.216

# autolabor
* ssh root@192.168.2.1
	* key: autolabor

# nvidia
* ssh nvidia@10.42.0.43

# udev rules
* lsusb
> ID `0403:6001` => (idVendor:idProduct)

* udevadm info -q all -n /dev/ttyUSB0 | grep ID_VENDOR_ID
> ID_VENDOR_ID=0403
 udevadm info -q all -n /dev/ttyUSB0 | grep ID_MODEL_ID
> ID_MODEL_ID=6001
```
# set the udev rule , make the device_port be fixed autolabor_pro1
#
KERNEL=="ttyUSB*", ATTRS{idVendor}=="0403", ATTRS{idProduct}=="6001", MODE:="0666", SYMLINK+="autolabor_pro1"
```

# tx2
sudo apt-get install ros-kinetic-dwa-local-planner
sudo apt-get install ros-kinetic-global-planner

# tx2_33
ssh nvidia@10.42.0.74

# 2018.09.01
## 33_tx2_vncviewer
wlan: 192.168.1.102
eth0: 10.42.0.1

# install x11vnc
https://blog.csdn.net/longhr/article/details/51657610
https://blog.csdn.net/styshoo/article/details/52706138
