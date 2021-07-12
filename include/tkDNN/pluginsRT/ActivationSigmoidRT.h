#include<cassert>
#include "../kernels.h"

class ActivationSigmoidRT : public IPluginV2 {

public:
	ActivationSigmoidRT() {


	}

	~ActivationSigmoidRT(){

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
		activationSIGMOIDForward((dnnType*)reinterpret_cast<const dnnType*>(inputs[0]), 
											reinterpret_cast<dnnType*>(outputs[0]), batchSize*size, stream);
		return 0;
	}
	#else
	virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override {

		activationSIGMOIDForward((dnnType*)reinterpret_cast<const dnnType*>(inputs[0]), 
											reinterpret_cast<dnnType*>(outputs[0]), batchSize*size, stream);
		return 0;
	}
	#endif 


	virtual size_t getSerializationSize() const NOEXCEPT override {
		return 1*sizeof(int);
	}

	virtual void serialize(void* buffer) const NOEXCEPT override {
		char *buf = reinterpret_cast<char*>(buffer),*a=buf;
		tk::dnn::writeBUF(buf, size);
		assert(buf == a + getSerializationSize());
	}

    const char* getPluginType() const NOEXCEPT override {
        return "ActivationSignmoidRT_TKDNN";
    }

    const char* getPluginVersion() const NOEXCEPT override{
        return "1";
    }
    void destroy() NOEXCEPT override{
        delete this;
    }
    bool supportsFormat(DataType type,PluginFormat format) const NOEXCEPT override {
        return true;
    }
    const char* getPluginNamespace() const NOEXCEPT override{
        return mPluginNamespace;
    }

    void setPluginNamespace(const char* pluginNamespace) NOEXCEPT override{
        mPluginNamespace = pluginNamespace;
    }

    const char* mPluginNamespace;
	int size;
};
