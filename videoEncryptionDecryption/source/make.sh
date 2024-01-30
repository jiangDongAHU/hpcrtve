rm CMakeCache.txt
rm -rf CMakeFiles
rm cmake_install.cmake
rm Makefile 
rm demo
cmake -B .
cmake --build .
./demo

