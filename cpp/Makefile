all: infer_basic libviterbi.so

infer_basic: infer_basic.cc viterbi.cc stopwatch.h
	g++ -o $@ $^ -lonnxruntime -O3 -Wall -Wextra -g

libviterbi.so: viterbi.cc
	g++ -o $@ $^ -shared -fPIC -O3 -Wall -Wextra -g

.PHONY: clean
clean:
	rm -f infer_basic libviterbi.so
