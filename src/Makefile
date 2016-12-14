
all: ghmock.o
	gcc ghmock.o -lm -lgsl -lgslcblas -lgomp -o ghmock

ghmock.o: ghmock.c
	gcc -O4 -fopenmp -DSTANDALONE -c ghmock.c

python: 
	python setup.py build_ext --inplace

clean:
	rm -f *.o
	rm -f ghost.so
	rm -rf build/
