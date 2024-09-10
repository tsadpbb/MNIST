const std = @import("std");

const fs = std.fs;
const heap = std.heap;
const meta = std.meta;
const math = std.math;
const mem = std.mem;
const crypto = std.crypto;
const fmt = std.fmt;

const endian = std.builtin.Endian.big;

const Allocator = mem.Allocator;
const File = std.fs.File;

const ERROR = 1;
const OKAY = 0;

const stdErrWriter = std.io.getStdErr().writer();
const stdOutWriter = std.io.getStdOut().writer();
const stdInReader = std.io.getStdIn().reader();

const print = std.debug.print;

//////////////////////////////
// Because I'm a brainlet   //
//////////////////////////////
//         X by Y           //
//            Y             //
//       [           ]      //
//    X  [           ]      //
//       [           ]      //
//                          //
//////////////////////////////
//                          //
//           Y              //
//       {                  //
//        {       },        //
//    X   {       },        //
//        {       },        //
//       }                  //
//                          //
//////////////////////////////
//                          //
//         [X][Y]T          //
//                          //
//////////////////////////////
fn Matrix(comptime T: type) type {
    return struct {
        const Self = @This();
        x: usize,
        y: usize,
        data: []T,
        allocator: Allocator,

        pub fn init(allocator: Allocator, x: usize, y: usize) !Self {
            const data: []T = try allocator.alloc(T, x*y);
            return .{
                .x = x,
                .y = y,
                .data = data,
                .allocator = allocator,
            };
        }

        pub fn get(self: *Self, x: usize, y: usize) T {
            return self.data[self.y*x+y];
        }

        pub fn set(self: *Self, x: usize, y: usize, new: T) T {
            const index = self.y*x+y;
            const old = self.data[index];
            self.data[index] = new;
            return old;
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.data);
            self.* = undefined;
        }

        /// Using this means you have to free the data yourself
        pub fn deinitNoFree(self: *Self) void {
            self.* = undefined;
        }

        /// Subtract another matrix from this one
        pub fn subtract(self: *Self, matrixPtr: *Matrix(T)) !void {
            var matrix = matrixPtr.*;
            if (self.x != matrix.x or self.y != matrix.y) {
                return error.InvalidMatSubtract;
            }
            for (0..self.x) |x| {
                for (0..self.y) |y| {
                    const value = self.get(x, y) - matrix.get(x, y);
                    _ = self.set(x, y, value);
                }
            }
        }

        pub fn multiplyScalar(self: *Self, scalar: T) void {
            for (0..self.data.len) |index| {
                self.data[index] = self.data[index] * scalar;
            }
        }

        pub fn hadamard(self: *Self, matrixPtr: *Matrix(T)) !void {
            var matrix = matrixPtr.*;
            if (self.x != matrix.x or self.y != matrix.y) {
                return error.InvalidMatSubtract;
            }
            for (0..self.x) |x| {
                for (0..self.y) |y| {
                    const value = self.get(x, y) * matrix.get(x, y);
                    _ = self.set(x, y, value);
                }
            }
        }

        /// Add a vector along each x or y index of the matrix
        pub fn addVector(self: *Self, vector: []T, dimension: MatrixDimension) !void {
            if (dimension == MatrixDimension.column) {
                if (vector.len != self.x) {
                    return error.InvalidVectorSize;
                }
                for (0..vector.len) |x| {
                    for(0..self.y) |y| {
                        const value = self.get(x, y) + vector[x];
                        _ = self.set(x, y, value);
                    }
                }
            } else {
                if (vector.len != self.y) {
                    return error.InvalidVectorSize;
                }
                for (0..vector.len) |y| {
                    for(0..self.x) |x| {
                        const value = self.get(x, y) + vector[y];
                        _ = self.set(x, y, value);
                    }
                }
            }
        }

        pub fn applyFn(self: *Self, function: fn (comptime T: type, x: T) T) void {
            for (0..self.data.len) |index| {
                self.data[index] = function(T, self.data[index]);
            }
        }

        pub fn debugPrint(self: *Self, comptime name: []const u8) void {
            print("\n{s}:\n\tMatrix Size: x: {d}, y: {d}\n\tMatrix Length: {d}\n", .{name, self.x, self.y, self.data.len});
        }

        pub fn debugPrintWithData(self: *Self, comptime name: []const u8) void {
            print("\n{s}:\n\tMatrix Size: x: {d}, y: {d}\n\tMatrix Length: {d}\n\tMatrix Data: {d}\n", .{name, self.x, self.y, self.data.len, self.data});
        }
    };
}

fn Layer(comptime T: type) type {
    return struct {
        const Self = @This();
        weights: Matrix(T), // Matrix
        biases: []T, // Vector
        allocator: Allocator,

        pub fn init(allocator: Allocator, previousLayerSize: usize, layerSize: usize ) !Self {
            const weights = try Matrix(T).init(allocator, layerSize, previousLayerSize);
            for(0..weights.data.len) |index| {
                weights.data[index] = crypto.random.float(T);
            }

            const biases = try allocator.alloc(T, layerSize);
            for(0..biases.len) |index| {
                biases[index] = crypto.random.float(T);
            }

            return .{
                .weights = weights,
                .biases = biases,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.weights.deinit();
            self.allocator.free(self.biases);
            self.* = undefined;
        }

        /// Using this means you are freeing the memory yourself
        pub fn deintNoFree(self: *Self) void {
            self.weights.deinitNoFree();
            self.* = undefined;
        }
    };
}

/// T being the type used in the weights, I1T being the T of IDX1
fn Network(comptime T: type, comptime I1T: type) type {
    return struct {
        const Self = @This();
        layers: []Layer(T),
        filename: ?[]u8,
        allocator: Allocator,

        /// Initializes a Network struct with random weights and biases
        pub fn init(allocator: Allocator, layerSizes: []usize) !Self {
            const layers = try allocator.alloc(Layer(T), layerSizes.len-1);

            for (layerSizes[0..layerSizes.len-1], layerSizes[1..layerSizes.len], 0..layers.len) |prev, layer, index| {
                layers[index] = try Layer(T).init(allocator, prev, layer);
            }

            return .{
                .layers = layers,
                .filename = null,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            for (0..self.layers.len) |index| {
                // This was read so we have to align the data to match the alloc alignment
                if(self.filename) |_| {
                    var layer = self.layers[index];
                    const weightData = layer.weights.data;
                    const biases = layer.biases;
                    
                    const rawWeightData: []u8 = @alignCast(mem.sliceAsBytes(weightData));
                    const rawBiases: []u8 = @alignCast(mem.sliceAsBytes(biases));

                    self.allocator.free(rawWeightData);
                    self.allocator.free(rawBiases);

                    layer.weights.deinitNoFree();
                    layer.deintNoFree();
                } else {
                    var layer = self.layers[index];
                    layer.deinit();
                }
            }
            self.allocator.free(self.layers);
            self.* = undefined;
        }

        /// Loads a Network from a specified file
        pub fn read(allocator: Allocator, filename: []u8) !Self {
            const cwd = fs.cwd();
            const file = try cwd.openFile(filename, .{});
            defer file.close();
            try file.seekTo(0);

            const reader = file.reader();

            // Read the magic number
            const zero1 = try reader.readByte();
            const zero2 = try reader.readByte();
            if (zero1 != 0 or zero2 != 0) {
                return error.InvalidFile;
            }

            const typeSize = try reader.readByte();
            if(typeSize != @sizeOf(T)) {
                return error.InvalidTypeSize;
            }

            const layerCount = try reader.readInt(usize, endian);
            const layers: []Layer(T) = try allocator.alloc(Layer(T), layerCount);
            const layerSizes: []usize = try allocator.alloc(usize, layerCount+1);
            defer allocator.free(layerSizes);
            errdefer allocator.free(layers);

            for(0..layerSizes.len) |index| {
                layerSizes[index] = try reader.readInt(usize, endian);
            }

            for (layerSizes[0..layerSizes.len-1], layerSizes[1..layerSizes.len], 0..layers.len) |prev, current, index| {
                // Read in raw bytes
                const rawWeightData: []u8 = try allocator.alloc(u8, prev*current*@sizeOf(T));
                _ = try reader.readAtLeast(rawWeightData, prev*current*@sizeOf(T));

                const rawBiases: []u8 = try allocator.alloc(u8, current*@sizeOf(T));
                _ = try reader.readAtLeast(rawBiases, current*@sizeOf(T));

                // Cast to correct type
                const weightData: []T = @alignCast(mem.bytesAsSlice(T, rawWeightData));
                const biases: []T = @alignCast(mem.bytesAsSlice(T, rawBiases));

                const weights: Matrix(T) = .{
                    .x = current,
                    .y = prev,
                    .data = weightData,
                    .allocator = allocator,
                };
                const layer: Layer(T) = .{
                    .weights = weights,
                    .biases = biases,
                    .allocator = allocator,
                };
                layers[index] = layer;
            }

            return .{
                .layers = layers,
                .filename = filename,
                .allocator = allocator,
            };
        }

        pub fn write(self: *Self, filename: []u8) !void {
            const cwd = fs.cwd();
            const file = try cwd.createFile(filename, .{});
            defer file.close();
            try file.seekTo(0);

            const writer = file.writer();

            // First two bytes are zero
            // third is the size of the type used
            // Fourth (as usize) is the number of layers
            // Then we print the dimensions of the layers, something like 784 100 10
            // Then we say fiwb and write everything
            try writer.writeByte(0);
            try writer.writeByte(0);
            try writer.writeByte(@sizeOf(T));
            try writer.writeInt(usize, self.layers.len, endian);
            for(self.layers) |layer| {
                try writer.writeInt(usize, layer.weights.y, endian);
            }
            // Don't forget the last layer
            try writer.writeInt(usize, self.layers[self.layers.len-1].weights.x, endian);
            for(self.layers) |layer| {
                // Convert to bytes
                const rawWeights: []u8 = mem.sliceAsBytes(layer.weights.data);
                const rawBiases: []u8 = mem.sliceAsBytes(layer.biases);
                try writer.writeAll(rawWeights);
                try writer.writeAll(rawBiases);
            }
        }

        /// Will take in a matrix of all the images you want to run and output an array
        /// You will have to find the max float in the array to determine what the model "thinks" the number is
        pub fn inference(self: *Self, matrix: Matrix(T)) !Matrix(T) {
            var current = matrix;

            for(self.layers, 0..self.layers.len) |layer, index| {
                var weights = layer.weights;
                const biases = layer.biases;

                // Forward pass
                var activation = try MatMul(T, &weights, &current);
                try activation.addVector(biases, MatrixDimension.column);
                activation.applyFn(sigmoid);

                if(index != 0) {
                    current.deinit();
                }
                current = activation;
            }
            return current;
        }

        /// Prepares the batches and runs the epoch
        /// Remeber that data is transposed to what you want
        pub fn train(self: *Self, inputsPtr: *Matrix(T), labels: []I1T, batchSize: usize, learningRate: T) !void {
            var inputs = inputsPtr.*;
            var rounds = inputs.x / batchSize;
            const leftovers = inputs.x % batchSize;

            if(leftovers > 0) rounds += 1;

            var indices = try self.allocator.alloc(usize, inputs.x);
            for (0..inputs.x) |i| {
                indices[i] = i;
            }

            for (0..rounds) |round| {
                if (round+1 == rounds) {
                    print("\tBatch: {d} of {d}\n", .{round+1, rounds});
                } else {
                    print("\tBatch: {d} of {d}\r", .{round+1, rounds});
                }
                var size = batchSize;
                // If it's the last round
                if (round == rounds-1 and leftovers > 0) {
                    size = leftovers;
                }
                var batch = try Matrix(T).init(self.allocator, inputs.y, size);
                var label = try Matrix(T).init(self.allocator, self.layers[self.layers.len-1].biases.len, size);

                defer {
                    batch.deinit();
                    label.deinit();
                }

                for (0..size) |y| {
                    const indicesIndex = crypto.random.intRangeAtMost(usize, 0, indices.len-1);
                    const index = try getAndRemove(self.allocator, &indices, indicesIndex);
                    const answer = labels[index];
                    for (0..batch.x) |x| {
                        _ = batch.set(x, y, inputs.get(index, x));
                    }
                    for (0..label.x) |cx| {
                        if (answer == cx) {
                            _ = label.set(cx, y, 1);
                        } else {
                            _ = label.set(cx, y, 0);
                        }
                    }
                }
                try self.trainBatch(&batch, &label, learningRate);
            }
        }

        /// Trains an indiviudal batch
        pub fn trainBatch(self: *Self, batchPtr: *Matrix(T), labelsPtr: *Matrix(T), learningRate: T) !void {
            const batch = batchPtr.*;
            const numberOfElements = batch.y;
            var labels = labelsPtr.*;

            // We have to allocate arrays to cache values, I think we just need to cache activation and error arrays
            var activations = try self.allocator.alloc(Matrix(T), self.layers.len);
            var errors = try self.allocator.alloc(Matrix(T), self.layers.len);
            defer {
                self.allocator.free(activations);
                self.allocator.free(errors);
            }
            errdefer {
                for (0..self.layers.len) |allIndex| {
                    errors[allIndex].deinit();
                    activations[allIndex].deinit();
                }
            }

            var current = batch;

            for(self.layers, 0..self.layers.len) |layer, index| {
                var weights = layer.weights;
                const biases = layer.biases;

                // Forward pass
                var activation = try MatMul(T, &weights, &current);
                try activation.addVector(biases, MatrixDimension.column);
                activation.applyFn(sigmoid);

                activations[index] = activation;
                current = activation;
            }

            // Alright, now its time to build the error array
            errors[errors.len-1] = try MatSubtract(T, &current, &labels);

            if (errors.len > 1) {
                var errorIndex = errors.len-2;
                while (errorIndex >= 0) : (errorIndex -= 1) {
                    var layerError = try applyFn(T, &activations[errorIndex], sigmoidDerivativeOnActivation);

                    var nextLayerWeights = self.layers[errorIndex+1].weights;
                    var nextLayerError = errors[errorIndex+1];

                    var nextLayerWeightsT = try transpose(T, &nextLayerWeights);
                    defer nextLayerWeightsT.deinit();

                    var interim = try MatMul(T, &nextLayerWeightsT, &nextLayerError);
                    defer interim.deinit();

                    try layerError.hadamard(&interim);

                    errors[errorIndex] = layerError;

                    if(errorIndex == 0) {
                        break;
                    }
                }
            }

            // Now that we've collected all the errors, time to adjust the weights and biases accordingly
            for(self.layers, 0..self.layers.len) |layer, index| {
                var weights = layer.weights;
                var biases = layer.biases;
                var layerError: Matrix(T) = errors[index];
                var layerActivation: Matrix(T) = undefined;
                if (index == 0) {
                    layerActivation = batchPtr.*;
                } else {
                    layerActivation = activations[index-1];
                }

                var layerActivationT = try transpose(T, &layerActivation);
                defer layerActivationT.deinit();

                var weightInterim = try MatMul(T, &layerError, &layerActivationT);
                defer weightInterim.deinit();

                weightInterim.multiplyScalar(learningRate / @as(T, @floatFromInt(numberOfElements)));

                try weights.subtract(&weightInterim);

                var summedError = try collapse(T, &layerError, MatrixDimension.row);
                defer summedError.deinit();

                summedError.multiplyScalar(learningRate / @as(T, @floatFromInt(numberOfElements)));

                for (0..biases.len) |i| {
                    biases[i] -= summedError.data[i];
                }
            }

            for (0..self.layers.len) |allIndex| {
                errors[allIndex].deinit();
                activations[allIndex].deinit();
            }
        }
    };
}

fn getAndRemove(allocator: Allocator, arrayPtr: *[]usize, index: usize) !usize {
    const array = arrayPtr.*;
    if (index >= array.len) {
        return error.IndexOutOfBounds;
    }
    if (array.len == 0) {
        return error.EmptyArray;
    }
    const value = array[index];
    const last = array[array.len-1];

    array[index] = last;
    arrayPtr.* = try allocator.realloc(array, array.len-1);

    return value;
}


// Thanks Chad Gippity
fn castToType(T: type, value: anytype) T {
    if (@typeInfo(T) == .Int) {
        if (@typeInfo(@TypeOf(value)) == .Float) {
            return @intFromFloat(value);
        } else {
            return @intCast(value);
        }
    } else if (@typeInfo(T) == .Float) {
        if (@typeInfo(@TypeOf(value)) == .Int) {
            return @floatFromInt(value);
        } else {
            return @floatCast(value);
        }
    } else {
        @panic("Unsupported type");
    }
}

/// Creates a transpose of the supplied matrix.
/// Remember to deinit!
fn transpose(comptime T: type, matrixPtr: *Matrix(T)) !Matrix(T) {
    var matrix = matrixPtr.*;
    var newMatrix = try Matrix(T).init(matrix.allocator, matrix.y, matrix.x);
    for(0..matrix.x) |x| {
        for (0..matrix.y) |y| {
            _ = newMatrix.set(y, x, matrix.get(x, y));
        }
    }
    return newMatrix;
}


const MatrixDimension = enum { 
    row, 
    column, 
};

/// Creates a matrix (of size x by 1 or 1 by y) of the supplied matrix.
/// Sums all values along a row or column
/// Remember to deinit!
fn collapse(comptime T: type, matrixPtr: *Matrix(T), dimension: MatrixDimension) !Matrix(T) {
    var matrix = matrixPtr.*;

    var newMatrix: Matrix(T) = undefined;
    if (dimension == MatrixDimension.row) {
        newMatrix = try Matrix(T).init(matrix.allocator, matrix.x, 1);
        @memset(newMatrix.data, 0);
        for (0..matrix.x) |x| {
            for (0..matrix.y) |y| {
                newMatrix.data[x] += matrix.get(x, y);
            }
        }
    } else {
        newMatrix = try Matrix(T).init(matrix.allocator, 1, matrix.y);
        @memset(newMatrix.data, 0);
        for (0..matrix.x) |x| {
            for (0..matrix.y) |y| {
                newMatrix.data[y] += matrix.get(x, y);
            }
        }
    }
    return newMatrix;
}

/// Creates a matrix that is one - two.
/// Remember to deinit!
fn MatSubtract(comptime T: type, onePtr: *Matrix(T), twoPtr: *Matrix(T)) !Matrix(T) {
    var one = onePtr.*;
    var two = twoPtr.*;
    var newMatrix = try Matrix(T).init(one.allocator, one.x, two.y);
    for (0..one.x) |x| {
        for (0..two.y) |y| {
            const value = one.get(x, y) - two.get(x, y);
            _ = newMatrix.set(x, y, value);
        }
    }
    return newMatrix;
}

/// Matrix multiplication, one * two
/// Remember to deinit!
fn MatMul(comptime T: type, onePtr: *Matrix(T), twoPtr: *Matrix(T)) !Matrix(T) {
    var one = onePtr.*;
    var two = twoPtr.*;
    var newMatrix = try Matrix(T).init(one.allocator, one.x, two.y);
    @memset(newMatrix.data, 0);
    for (0..one.x) |a| {
        for (0..two.y) |b| {
            for (0..two.x, 0..one.y) |c, d| {
                newMatrix.data[two.y*a+b] += one.get(a, d) * two.get(c, b);
            }
        }
    }
    return newMatrix;
}

/// Create a new matrix that consists of the elements of the supplied matrix put into the supplied function.
/// Remember to deinit!
fn applyFn(comptime T: type, matrixPtr: *Matrix(T), function: fn (comptime T: type, x: T) T) !Matrix(T) {
    const matrix = matrixPtr.*;
    var newMatrix = try Matrix(T).init(matrix.allocator, matrix.x, matrix.y);
    for (0..matrix.data.len) |index| {
        newMatrix.data[index] = function(T, matrix.data[index]);
    }
    return newMatrix;
}

/// Using this because derivative of sigmoid on z is just a*(1-a)
fn sigmoidDerivativeOnActivation(comptime T: type, x: T) T {
    return x*(1-x);
}

fn sigmoid(comptime T: type, x: T) T {
    return (1 / (1+@exp(-1*x)));
}

const IDXType = enum(u8) { 
    u8 = 8, 
    i8 = 9, 
    i16 = 11, 
    i32 = 12, 
    f32 = 13, 
    f64 = 14
};

fn IDXTypeSize (input: IDXType) usize {
    return switch(input) {
        IDXType.u8, IDXType.i8 => 1,
        IDXType.i16 => 2,
        IDXType.i32, IDXType.f32 => 4,
        IDXType.f64 => 8,
    };
}

fn IDX1(comptime T: type) type {
    return struct {
        const Self = @This();
        filename: []u8,
        dataType: IDXType,
        dimensions: u8,
        itemCount: u32,
        data: []T,
        allocator: Allocator,

        pub fn read(allocator: Allocator, filename: []u8) !Self {
            const cwd = fs.cwd();
            const file = try cwd.openFile(filename, .{});
            defer file.close();
            try file.seekTo(0);

            const reader = file.reader();

            // Read the magic number
            const zero1 = try reader.readByte();
            const zero2 = try reader.readByte();
            if (zero1 != 0 or zero2 != 0) {
                return error.InvalidFile;
            }

            const dataType: IDXType = @enumFromInt(try reader.readByte());
            if (IDXTypeSize(dataType) != @sizeOf(T)) {
                return error.IncorrectTypeSize;
            }

            // I recognize you are supposed to use this to determine how many dimension bytes to read, but I don't care
            const dimensions = try reader.readByte();

            // Read the sizes
            const itemCount = try reader.readInt(u32, endian);

            const raw = try reader.readAllAlloc(allocator, itemCount*@sizeOf(T));
            const data: []T = @alignCast(mem.bytesAsSlice(T, raw));

            if(data.len != itemCount) {
                allocator.free(data);
                return error.FileTooSmallError;
            }

            return .{ 
                .filename = filename,
                .dataType = dataType, 
                .dimensions = dimensions, 
                .itemCount = itemCount, 
                .data = data, 
                .allocator = allocator
            };
        }

        pub fn deinit(self: *Self) void {
            const raw: []u8 = @alignCast(mem.sliceAsBytes(self.data));
            self.allocator.free(raw);
            self.* = undefined;
        }
    };
}

fn IDX3(comptime T: type, comptime FT: type) type {
    return struct {
        const Self = @This();
        filename: []u8,
        dataType: IDXType,
        dimensions: u8,
        itemCount: u32,
        rowCount: u32,
        columnCount: u32,
        data: Matrix(FT),
        allocator: Allocator,

        pub fn read(allocator: Allocator, filename: []u8) !Self {
            const cwd = fs.cwd();
            const file = try cwd.openFile(filename, .{});
            defer file.close();
            try file.seekTo(0);

            const reader = file.reader();

            // Read the magic number
            const zero1 = try reader.readByte();
            const zero2 = try reader.readByte();
            if (zero1 != 0 or zero2 != 0) {
                return error.InvalidFile;
            }

            const dataType: IDXType = @enumFromInt(try reader.readByte());
            if (IDXTypeSize(dataType) != @sizeOf(T)) {
                return error.IncorrectTypeSize;
            }

            // I recognize you are supposed to use this to determine how many dimension bytes to read, but I don't care
            const dimensions = try reader.readByte();

            // Read the sizes
            const itemCount = try reader.readInt(u32, endian);
            const rowCount = try reader.readInt(u32, endian);
            const columnCount = try reader.readInt(u32, endian);

            const size = itemCount*rowCount*columnCount;

            const raw = try reader.readAllAlloc(allocator, size*@sizeOf(T));
            const data: []T = @alignCast(mem.bytesAsSlice(T, raw));

            if(data.len != size) {
                allocator.free(data);
                return error.FileTooSmallError;
            }

            // Convert to the type specified in Network
            const floatData = try allocator.alloc(FT, data.len);
            for (0..data.len) |index| {
                floatData[index] = castToType(FT, data[index]);
            }

            // While we are here, let's normalize
            const length = rowCount*columnCount;
            for (0..itemCount) |itemNumber| {
                var total: FT = 0;
                for (0..length) |index| {
                    const value = floatData[length*itemNumber+index];
                    total += value*value;
                }
                const magnitude = @sqrt(total);
                for (0..length) |index| {
                    const value: FT = floatData[length*itemNumber+index];
                    const newvalue: FT = value / magnitude;
                    floatData[length*itemNumber+index] = newvalue;
                }
            }

            allocator.free(raw);

            const dataMatrix: Matrix(FT) = .{
                .x = itemCount,
                .y = rowCount*columnCount,
                .data = floatData,
                .allocator = allocator,
            };

            return .{ 
                .filename = filename,
                .dataType = dataType, 
                .dimensions = dimensions, 
                .itemCount = itemCount, 
                .rowCount = rowCount, 
                .columnCount = columnCount, 
                .data = dataMatrix, 
                .allocator = allocator
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.data.data);
            self.data.deinitNoFree();
            self.* = undefined;
        }

        pub fn debugPrint(self: *Self) void {
            print("IDX3 Filename: {s}\nData Type: {s}\nDimensions: {d}\nImage Count: {d}\nRow Count: {d}\nColumn Count: {d}\nData Size: {d}\n", .{self.filename, @tagName(self.dataType), self.dimensions, self.imageCount, self.rowCount, self.columnCount, self.data.len});
        }
    };
}

const ProgramMode = enum {
    Train,
    Inference
};

// https://yann.lecun.com/exdb/mnist/ refer to this to read IDX files
pub fn main() !u8 {
    var gpa = heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer {
        const status = gpa.deinit();
        if (status == .leak) {
            @panic("Leak Detected");
        }
    }

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        _ = try stdErrWriter.write("Not enound arguments supplied!\n");
        return ERROR;
    }

    var mode: ?ProgramMode = null;
    var IDX3Filename: ?[]u8 = null;
    var IDX1Filename: ?[]u8 = null;
    var modelFilename: ?[]u8 = null;
    var outputFilename: ?[]u8 = null;
    var layerString: ?[]u8 = null;
    var batchSize: ?usize = null;
    var epochs: ?usize = null;
    var learningRate: ?f32 = null;

    // The first argument is the program
    // Check all the stupid arguments why are there so many
    var index: usize = 1;
    while (index < args.len) {
        const arg = args[index];
        if(mem.eql(u8, arg, "-t")) {
            if (mode) |_| {
                _ = try stdErrWriter.write("Program mode specified more than once\n");
                return ERROR;
            }
            mode = ProgramMode.Train;
        } else if(mem.eql(u8, arg, "-i")) {
            if (mode) |_| {
                _ = try stdErrWriter.write("Program mode specified more than once\n");
                return ERROR;
            }
            mode = ProgramMode.Inference;
        } else if (mem.eql(u8, arg, "-f3")) {
            if (args.len-1 < index+1) {
                _ = try stdErrWriter.write("IDX3 filename not supplied!\n");
                return ERROR;
            }
            if (IDX3Filename) |_| {
                _ = try stdErrWriter.write("IDX3 filename specified more than once\n");
                return ERROR;
            }
            IDX3Filename = args[index+1];
            index += 2;
            continue;
        } else if (mem.eql(u8, arg, "-f1")) {
            if (args.len-1 < index+1) {
                _ = try stdErrWriter.write("IDX1 filename not supplied!\n");
                return ERROR;
            }
            if (IDX1Filename) |_| {
                _ = try stdErrWriter.write("IDX1 filename specified more than once\n");
                return ERROR;
            }
            IDX1Filename = args[index+1];
            index += 2;
            continue;
        } else if (mem.eql(u8, arg, "-m")) {
            if (args.len-1 < index+1) {
                _ = try stdErrWriter.write("Model filename not supplied!\n");
                return ERROR;
            }
            if (modelFilename) |_| {
                _ = try stdErrWriter.write("Model filename specified more than once\n");
                return ERROR;
            }
            modelFilename = args[index+1];
            index += 2;
            continue;
        } else if (mem.eql(u8, arg, "-o")) {
            if (args.len-1 < index+1) {
                _ = try stdErrWriter.write("Output filename not supplied!\n");
                return ERROR;
            }
            if (outputFilename) |_| {
                _ = try stdErrWriter.write("Output filename specified more than once!\n");
                return ERROR;
            }
            outputFilename = args[index+1];
            index += 2;
            continue;
        } else if (mem.eql(u8, arg, "-l")) {
            if (args.len-1 < index+1) {
                _ = try stdErrWriter.write("Layer format not supplied!\n");
                return ERROR;
            }
            if (layerString) |_| {
                _ = try stdErrWriter.write("Layer format specified more than once!\n");
                return ERROR;
            }
            layerString = args[index+1];
            index += 2;
            continue;
        } else if (mem.eql(u8, arg, "-b")) {
            if (args.len-1 < index+1) {
                _ = try stdErrWriter.write("Batch size not supplied!\n");
                return ERROR;
            }
            if (batchSize) |_| {
                _ = try stdErrWriter.write("Batch size specified more than once!\n");
                return ERROR;
            }
            const batchString = args[index+1];
            batchSize = try fmt.parseInt(usize, batchString, 10);
            index += 2;
            continue;
        } else if (mem.eql(u8, arg, "-e")) {
            if (args.len-1 < index+1) {
                _ = try stdErrWriter.write("Epochs not supplied!\n");
                return ERROR;
            }
            if (epochs) |_| {
                _ = try stdErrWriter.write("Epochs specified more than once!\n");
                return ERROR;
            }
            const epochString = args[index+1];
            epochs = try fmt.parseInt(usize, epochString, 10);
            index += 2;
            continue;
        } else if (mem.eql(u8, arg, "-r")) {
            if (args.len-1 < index+1) {
                _ = try stdErrWriter.write("Learning rate not supplied!\n");
                return ERROR;
            }
            if (learningRate) |_| {
                _ = try stdErrWriter.write("Learning rate specified more than once!\n");
                return ERROR;
            }
            const rateString = args[index+1];
            learningRate = try fmt.parseFloat(f32, rateString);
            index += 2;
            continue;
        }
        else {
            _ = try stdErrWriter.write("Invalid Argument\n");
            return ERROR;
        }
        index += 1;
    }


    if (mode == null) {
        _ = try stdErrWriter.write("You have to specify if you want to do training or do inference!\n");
        return ERROR;
    }

    if (IDX1Filename == null) {
        _ = try stdErrWriter.write("You have to specify an IDX1 filename!\n");
        return ERROR;
    }

    if (IDX3Filename == null) {
        _ = try stdErrWriter.write("You have to specify an IDX3 filename!\n");
        return ERROR;
    }

    if (mode == ProgramMode.Inference and modelFilename == null) {
        _ = try stdErrWriter.write("You have to specify a model filename to load for inference!\n");
        return ERROR;
    }

    if (mode == ProgramMode.Train and outputFilename == null) {
        _ = try stdErrWriter.write("You have to specify an output filename to save to after training!\n");
        return ERROR;
    }

    if (modelFilename != null and layerString != null) {
        _ = try stdErrWriter.write("You can't specify a layer format to initialize and a model name to load at the same time.\n");
        return ERROR;
    }

    if (mode == ProgramMode.Train and batchSize == null) {
        _ = try stdErrWriter.write("You need to specify a batch size when training a model.\n");
        return ERROR;
    }

    if (mode == ProgramMode.Inference and batchSize != null) {
        _ = try stdErrWriter.write("You can't specify a batch size when doing inference on a model.\n");
        return ERROR;
    }

    if (mode == ProgramMode.Train and epochs == null) {
        _ = try stdErrWriter.write("You need to specify the number of epochs when training a model.\n");
        return ERROR;
    }

    if (mode == ProgramMode.Inference and epochs != null) {
        _ = try stdErrWriter.write("You can't specify the number of epochs when doing inference on a model.\n");
        return ERROR;
    }

    if (mode == ProgramMode.Train and learningRate == null) {
        _ = try stdErrWriter.write("You need to specify a learning when training a model.\n");
        return ERROR;
    }

    if (mode == ProgramMode.Inference and batchSize != null) {
        _ = try stdErrWriter.write("You can't specify a learning rate when doing inference on a model.\n");
        return ERROR;
    }

    var IDX3File = try IDX3(u8, f32).read(allocator, IDX3Filename.?);
    defer IDX3File.deinit();

    var IDX1File = try IDX1(u8).read(allocator, IDX1Filename.?);
    defer IDX1File.deinit();

    var layerSize = try allocator.alloc(usize, 2);
    defer allocator.free(layerSize);

    layerSize[0] = IDX3File.rowCount*IDX3File.columnCount;
    if(layerString) |string| {
        layerSize = try allocator.realloc(layerSize, layerString.?.len+2);

        var iter = mem.tokenizeScalar(u8, string, ',');
        var layerNumber: usize = 1;
        while(iter.next()) |num| {
            layerSize[layerNumber] = try fmt.parseInt(usize, num, 10);
            layerNumber += 1;
        }
        layerSize = try allocator.realloc(layerSize, layerNumber+1);
    }
    layerSize[layerSize.len-1] = 10;

    var network: Network(f32, u8) = undefined;
    if (modelFilename) |filename| {
        network = try Network(f32, u8).read(allocator, filename);
    } else {
        network = try Network(f32, u8).init(allocator, layerSize);
    }
    defer network.deinit();

    if (mode == ProgramMode.Train) {
        for (0..epochs.?) |epoch| {
            try stdOutWriter.print("Epoch: {d} of {d}\n", .{epoch+1, epochs.?});
            try network.train(&IDX3File.data, IDX1File.data, batchSize.?, learningRate.?);
        }
        try network.write(outputFilename.?);
    } else {
        // We have to transpose to go from 60000 x 784 to 784 x 60000, better for input to matrix in network
        var input = try transpose(f32, &IDX3File.data);

        var out = try network.inference(input);

        // Transpose again to bring it back to 60000 x 10
        var outT = try transpose(f32, &out);
        out.deinit(); 

        var numCorrect: usize = 0;
        for(0..outT.x, IDX1File.data) |outputIndex, correct| {
            const output = outT.data[outT.y*outputIndex..outT.y*(outputIndex+1)];
            var maxIndex: usize = 0;
            for(0..output.len) |i| {
                if (output[i] > output[maxIndex]) {
                    maxIndex = i;
                }
            }

            if(maxIndex == @as(usize, correct)) {
                numCorrect += 1;
            } else {
                const matrix = IDX3File.data;
                const size = IDX3File.columnCount*IDX3File.rowCount;
                for (0..IDX3File.rowCount) |row| {
                    for (0..IDX3File.columnCount) |colummn| {
                        const value = matrix.data[size*outputIndex+IDX3File.rowCount*row+colummn];
                        if (value == 0) {
                            _ = try stdOutWriter.write("  ");
                        } else {
                            _ = try stdOutWriter.write("XX");
                        }
                    }
                    _ = try stdOutWriter.write("\n");
                }
                print("Incorrect Guess!\nOutput: {d}\nGuess: {d}\nAnswer: {d}\n", .{output, maxIndex, correct});
            }
        }

        try stdOutWriter.print("The Results are in! {d} / {d}\n", .{numCorrect, IDX3File.itemCount});

        outT.deinit();
        input.deinit();
    }

    return OKAY;
}
