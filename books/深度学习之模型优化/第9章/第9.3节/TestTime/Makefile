CC = g++
CXXFLAGS = -g -Wall -fPIC -std=c++11

INCS_HEAD += ../../build/install/include/ncnn
LIB_PATH += ../../build/install/lib/ /usr/local/lib/
LIBS:= -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_core -lncnn

simpleconv5:simpleconv5.o
	$(CC) simpleconv5.cpp -o simpleconv5 $(CXXFLAGS) $(addprefix -I,$(INCS_HEAD)) $(addprefix -L,$(LIB_PATH)) $(LIBS)
simpleconv5.o:simpleconv5.cpp
	$(CC) -c simpleconv5.cpp -o simpleconv5.o $(CXXFLAGS) $(addprefix -I,$(INCS_HEAD)) $(LIBS)
clean:
	rm *o


