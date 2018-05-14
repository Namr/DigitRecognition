SRC_FILES = $(wildcard src/*.cpp)
BUILD_FILES = $(patsubst src/%.cpp, build/%.o, ${SRC_FILES})
LIBS = 
LFLAGS = $(shell pkg-config --libs ${LIBS})
CFLAGS = -std=gnu++11 -g -O3 $(shell pkg-config --cflags ${LIBS})

all: build ${BUILD_FILES}
	g++ -o build/jellyVision ${BUILD_FILES} ${LFLAGS}
clean:
	-rm -rf build/
build/%.o: src/%.cpp
	g++ ${CFLAGS} -c -o $@ $^
build:
	mkdir build
