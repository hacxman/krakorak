g++ UEyeOpenCV.cpp -o ueye.o -c -I. -ggdb3
g++ main.cc -I. -std=c++11 -o main.o -c -ggdb3 -fopenmp
g++ main.o ueye.o -o a.out -lopencv_core -lopencv_video -lopencv_highgui -lopencv_imgproc -lopencv_flann -std=c++11 -I. -lueye_api
