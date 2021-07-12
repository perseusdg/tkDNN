#include<cassert>
#include "../kernels.h"

class ResizeLayerRT : public IPluginV2 {

public:
	ResizeLayerRT(int c, int h, int w) {
		o_c = c;
		o_h = h;
		o_w = w;	
	}

	~ResizeLayerRT(){
	}

	int getNbOutputs() const NOEXCEPT override {
		return 1;
	}

	Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) NOEXCEPT override {
		return Dims3{o_c, o_h, o_w};
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
		resizeForward((dnnType*)reinterpret_cast<const dnnType*>(inputs[0]), 
					  reinterpret_cast<dnnType*>(outputs[0]), 
					  batchSize, i_c, i_h, i_w, o_c, o_h, o_w, stream);
		return 0;	
	}
	#else
	virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override {
    	// printf("%d %d %d %d %d %d\n", i_c, i_w, i_h, o_c, o_w, o_h);
        resizeForward((dnnType*)reinterpret_cast<const dnnType*>(inputs[0]), 
					  reinterpret_cast<dnnType*>(outputs[0]), 
					  batchSize, i_c, i_h, i_w, o_c, o_h, o_w, stream);
		return 0;
	}
	#endif 


	virtual size_t getSerializationSize() const NOEXCEPT override {
		return 6*sizeof(int);
	}

	virtual void serialize(void* buffer) const NOEXCEPT override {
		char *buf = reinterpret_cast<char*>(buffer),*a=buf;

		tk::dnn::writeBUF(buf, o_c);
		tk::dnn::writeBUF(buf, o_h);
		tk::dnn::writeBUF(buf, o_w);

		tk::dnn::writeBUF(buf, i_c);
		tk::dnn::writeBUF(buf, i_h);
		tk::dnn::writeBUF(buf, i_w);
		assert(buf == a + getSerializationSize());
	}

    const char* getPluginType() const NOEXCEPT override {
        return "ResizeLayerRT_TKDNN";
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
        i_c = inputDims[0].d[0];
        i_h = inputDims[0].d[1];
        i_w = inputDims[0].d[2];
    }

    IPluginV2* clone() const NOEXCEPT override{
	    ResizeLayerRT* p = new ResizeLayerRT(*this);
	    p->setPluginNamespace(mPluginNamespace);
	    return p;
	}

	const char* mPluginNamespace;
	int i_c, i_h, i_w, o_c, o_h, o_w;
};
