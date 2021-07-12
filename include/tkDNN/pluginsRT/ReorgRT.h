#include<cassert>
#include "../kernels.h"

class ReorgRT : public IPluginV2 {

public:
	ReorgRT(int stride) {
		this->stride = stride;
	}

	~ReorgRT(){

	}

	int getNbOutputs() const NOEXCEPT override {
		return 1;
	}

	Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) NOEXCEPT override {
		return Dims3{inputs[0].d[0]*stride*stride, inputs[0].d[1]/stride, inputs[0].d[2]/stride};
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
		
		reorgForward((dnnType*)reinterpret_cast<const dnnType*>(inputs[0]), 
					  reinterpret_cast<dnnType*>(outputs[0]), 
					  batchSize, c, h, w, stride, stream);
		return 0;
	}
	#else
	virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override {

		reorgForward((dnnType*)reinterpret_cast<const dnnType*>(inputs[0]), 
					  reinterpret_cast<dnnType*>(outputs[0]), 
					  batchSize, c, h, w, stride, stream);
		return 0;
	}
	#endif 


	virtual size_t getSerializationSize() const NOEXCEPT override {
		return 4*sizeof(int);
	}

	virtual void serialize(void* buffer) const NOEXCEPT override {
		char *buf = reinterpret_cast<char*>(buffer),*a=buf;
		tk::dnn::writeBUF(buf, stride);
		tk::dnn::writeBUF(buf, c);
		tk::dnn::writeBUF(buf, h);
		tk::dnn::writeBUF(buf, w);
		assert(buf == a + getSerializationSize());
	}

    const char* getPluginType() const NOEXCEPT override {
        return "ReorgRT_TKDNN";
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

    void configureWithFormat(Dims const *inputDims,int32_t nbInputs,Dims const *outputDims,int32_t nbOutputs,DataType type,PluginFormat format,int32_t maxBatchSize) NOEXCEPT override{
        c = inputDims[0].d[0];
        h = inputDims[0].d[1];
        w = inputDims[0].d[2];
	}

	IPluginV2* clone() const NOEXCEPT override{
	    ReorgRT *p = new ReorgRT(*this);
	    p->setPluginNamespace(mPluginNamespace);
	    return p;
	}

    const char* mPluginNamespace;
	int c, h, w, stride;
};
