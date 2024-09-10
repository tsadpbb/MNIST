main:
	zig build-exe --name MNIST MNIST.zig

get-data:
	mkdir data && cd data
	wget https://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz && gunzip train-images-idx3-ubyte.gz

train:
	./MNIST -t -f3 data/train-images.idx3-ubyte -f1 data/train-labels.idx1-ubyte -m model.byte -b 10 -e 30 -r 0.5 -l 100

inference:
	./MNIST -i -f3 data/t10k-images.idx3-ubyte -f1 data/t10k-labels.idx1-ubyte -m model.byte

clean:
	rm -rf MNIST MNIST.o .zig-cache
