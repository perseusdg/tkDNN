#include<cassert>
#include "../kernels.h"

class MaxPoolFixedSizeRT : public IPluginV2 {

public:
	MaxPoolFixedSizeRT(int c, int h, int w, int n, int strideH, int strideW, int winSize, int padding) {
		this->c = c;	
		this->h = h;
		this->w = w;
		this->n = n;
		this->stride_H = strideH;
		this->stride_W = strideW;
		this->winSize = winSize;
		this->padding = padding;
	}

	~MaxPoolFixedSizeRT(){
	}

	int getNbOutputs() const NOEXCEPT override {
		return 1;
	}

	Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) NOEXCEPT override {
		return Dims3{this->c, this->h, this->w};
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
		MaxPoolingForward(srcData, dstData, batchSize, this->c, this->h, this->w, this->stride_H, this->stride_W, this->winSize, this->padding, stream);
		return 0;	
	}
	#else
	virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override {

		//std::cout<<this->n<<"  "<<this->c<<"  "<<this->h<<"  "<<this->w<<"  "<<this->stride_H<<"  "<<this->stride_W<<"  "<<this->winSize<<"  "<<this->padding<<std::endl;
		dnnType *srcData = (dnnType*)reinterpret_cast<const dnnType*>(inputs[0]);
		dnnType *dstData = reinterpret_cast<dnnType*>(outputs[0]);
		MaxPoolingForward(srcData, dstData, batchSize, this->c, this->h, this->w, this->stride_H, this->stride_W, this->winSize, this->padding, stream);
		return 0;
	}
	#endif 

	virtual size_t getSerializationSize() const NOEXCEPT override {
		return 8*sizeof(int);
	}

	virtual void serialize(void* buffer) const NOEXCEPT override {
		char *buf = reinterpret_cast<char*>(buffer),*a=buf;

		tk::dnn::writeBUF(buf, this->c);
		tk::dnn::writeBUF(buf, this->h);
		tk::dnn::writeBUF(buf, this->w);
		tk::dnn::writeBUF(buf, this->n);
		tk::dnn::writeBUF(buf, this->stride_H);
		tk::dnn::writeBUF(buf, this->stride_W);
		tk::dnn::writeBUF(buf, this->winSize);
		tk::dnn::writeBUF(buf, this->padding);
		assert(buf == a + getSerializationSize());
	}

	int n, c, h, w;
	int stride_H, stride_W;
	int winSize;
	int padding;
};
