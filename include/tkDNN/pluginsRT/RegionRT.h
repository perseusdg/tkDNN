#include<cassert>
#include "../kernels.h"

class RegionRT : public IPluginV2 {

public:
	RegionRT(int classes, int coords, int num) {

		this->classes = classes;
		this->coords = coords;
		this->num = num;
	}

	~RegionRT(){

	}

	int getNbOutputs() const NOEXCEPT override {
		return 1;
	}

	Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) NOEXCEPT override {
		return inputs[0];
	}


	int initialize() NOEXCEPT override {

		return 0;
	}

	virtual void terminate() NOEXCEPT override {
	}

	virtual size_t getWorkspaceSize(int maxBatchSize) const NOEXCEPT override {
		return 0;
	}

	#if NV_TENSORRT_MAJOR >= 8
	virtual int32_t enqueue(int32_t batchSize,void const*const* inputs,void*const* outputs,void* workspace,cudaStream_t stream) NOEXCEPT override{


		dnnType *srcData = (dnnType*)reinterpret_cast<const dnnType*>(inputs[0]);
		dnnType *dstData = reinterpret_cast<dnnType*>(outputs[0]);

		checkCuda( cudaMemcpyAsync(dstData, srcData, batchSize*c*h*w*sizeof(dnnType), cudaMemcpyDeviceToDevice, stream));

		for (int b = 0; b < batchSize; ++b){
			for(int n = 0; n < num; ++n){
				int index = entry_index(b, n*w*h, 0);
				activationLOGISTICForward(srcData + index, dstData + index, 2*w*h, stream);
				
				index = entry_index(b, n*w*h, coords);
				activationLOGISTICForward(srcData + index, dstData + index, w*h, stream);
			}
		}

		//softmax start
		int index = entry_index(0, 0, coords + 1);
		softmaxForward(	srcData + index, classes, batchSize*num, 
						(c*h*w)/num, 
						w*h, 1, w*h, 1, dstData + index, stream);

		return 0;
	}
	#else
	virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override {

		dnnType *srcData = (dnnType*)reinterpret_cast<const dnnType*>(inputs[0]);
		dnnType *dstData = reinterpret_cast<dnnType*>(outputs[0]);

		checkCuda( cudaMemcpyAsync(dstData, srcData, batchSize*c*h*w*sizeof(dnnType), cudaMemcpyDeviceToDevice, stream));

		for (int b = 0; b < batchSize; ++b){
			for(int n = 0; n < num; ++n){
				int index = entry_index(b, n*w*h, 0);
				activationLOGISTICForward(srcData + index, dstData + index, 2*w*h, stream);
				
				index = entry_index(b, n*w*h, coords);
				activationLOGISTICForward(srcData + index, dstData + index, w*h, stream);
			}
		}

		//softmax start
		int index = entry_index(0, 0, coords + 1);
		softmaxForward(	srcData + index, classes, batchSize*num, 
						(c*h*w)/num, 
						w*h, 1, w*h, 1, dstData + index, stream);

		return 0;
	}
	#endif


	virtual size_t getSerializationSize() const NOEXCEPT override {
		return 6*sizeof(int);
	}

	virtual void serialize(void* buffer) const NOEXCEPT override {
		char *buf = reinterpret_cast<char*>(buffer),*a=buf;
		tk::dnn::writeBUF(buf, classes);
		tk::dnn::writeBUF(buf, coords);
		tk::dnn::writeBUF(buf, num);
		tk::dnn::writeBUF(buf, c);
		tk::dnn::writeBUF(buf, h);
		tk::dnn::writeBUF(buf, w);
		assert(buf == a + getSerializationSize());
	}

	int c, h, w;
    int classes, coords, num;

	int entry_index(int batch, int location, int entry) {
		int n =   location / (w*h);
		int loc = location % (w*h);
		return batch*c*h*w + n*w*h*(coords+classes+1) + entry*w*h + loc;
	}

};
