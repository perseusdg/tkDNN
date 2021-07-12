#include "Int8Calibrator.h"

Int8EntropyCalibrator::Int8EntropyCalibrator(BatchStream& stream, int firstBatch, 
                                             const std::string& calibTableFilePath,
                                             const std::string& inputBlobName,
                                             bool readCache): 
    mStream(stream), 
    mCalibTableFilePath(calibTableFilePath),
    mInputBlobName(inputBlobName.c_str()),
    mReadCache(readCache) {
    nvinfer1::Dims4 dims = mStream.getDims();
    mInputCount = mStream.getBatchSize() * dims.d[1] * dims.d[2] * dims.d[2];
    checkCuda(cudaMalloc(&mDeviceInput, mInputCount * sizeof(float)));
    mStream.reset(firstBatch);
}

bool Int8EntropyCalibrator::getBatch(void* bindings[], const char* names[], int nbBindings) NOEXCEPT {
    if (!mStream.next())
        return false;

    checkCuda(cudaMemcpy(mDeviceInput, mStream.getBatch(), mInputCount * sizeof(float), cudaMemcpyHostToDevice));
    assert(!strcmp(names[0], mInputBlobName.c_str()));
    bindings[0] = mDeviceInput;
    return true;
}

const void* Int8EntropyCalibrator::readCalibrationCache(size_t& length) NOEXCEPT {
    mCalibrationCache.clear();
    assert(!mCalibTableFilePath.empty());
    std::ifstream input(mCalibTableFilePath, std::ios::binary);
    input >> std::noskipws;
    input >> std::noskipws;
    if (mReadCache && input.good())
        std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(),
                  std::back_inserter(mCalibrationCache));

    length = mCalibrationCache.size();
    return length ? &mCalibrationCache[0] : nullptr;
}

void Int8EntropyCalibrator::writeCalibrationCache(const void* cache, size_t length) NOEXCEPT {
    assert(!mCalibTableFilePath.empty());
    std::ofstream output(mCalibTableFilePath, std::ios::binary);
    output.write(reinterpret_cast<const char*>(cache), length);
    output.close();
}