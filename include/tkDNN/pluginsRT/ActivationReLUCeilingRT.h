#include<cassert>
#include "../kernels.h"

class ActivationReLUCeiling : public IPluginV2 {

public:
	ActivationReLUCeiling(const float ceiling) {
		this->ceiling = ceiling;
	}

	~ActivationReLUCeiling(){

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
	virtual int32_t enqueue(int batchSize,void const*const* inputs,void *const* outputs,void* workspace,cudaStream_t stream) NOEXCEPT override{
		activationReLUCeilingForward((dnnType*)reinterpret_cast<const dnnType*>(inputs[0]), 
											reinterpret_cast<dnnType*>(outputs[0]), batchSize*size, ceiling, stream);
		return 0;

	}
	#else
	virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override {

		activationReLUCeilingForward((dnnType*)reinterpret_cast<const dnnType*>(inputs[0]), 
											reinterpret_cast<dnnType*>(outputs[0]), batchSize*size, ceiling, stream);
		return 0;
	}
	#endif 



	virtual size_t getSerializationSize() const NOEXCEPT override {
		return 1*sizeof(int) +  1*sizeof(float);
	}

	virtual void serialize(void* buffer) const NOEXCEPT override {
		char *buf = reinterpret_cast<char*>(buffer),*a=buf;
		tk::dnn::writeBUF(buf, ceiling);
		tk::dnn::writeBUF(buf, size);
		assert(buf = a + getSerializationSize());
		
	}

	int size;
	float ceiling;
};
