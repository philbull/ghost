
all: ghmock.o
	gcc ghmock.o -lm -lgsl -lgslcblas -lgomp -o ghmock

ghmock.o: ghmock.c
	gcc -O4 -fopenmp -c ghmock.c

clean:
	rm *.o
