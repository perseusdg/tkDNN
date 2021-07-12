#include<cassert>

class FlattenConcatRT : public IPluginV2 {

public:
	FlattenConcatRT() {
		stat = cublasCreate(&handle);
		if (stat != CUBLAS_STATUS_SUCCESS) {
			printf ("CUBLAS initialization failed\n");
			return;
  		}
	}

	~FlattenConcatRT(){

	}

	int getNbOutputs() const NOEXCEPT override {
		return 1;
	}

	Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) NOEXCEPT override {
		return Dims3{ inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2], 1, 1};
	}

	

	int initialize() NOEXCEPT override {
		return 0;
	}

	virtual void terminate() NOEXCEPT override {
		checkERROR(cublasDestroy(handle));
	}

	virtual size_t getWorkspaceSize(int maxBatchSize) const NOEXCEPT override {
		return 0;
	}

	#if NV_TENSORRT_MAJOR >= 8
	virtual int32_t enqueue(int32_t batchSize, void const*const* inputs,void*const* outputs,void* workspace,cudaStream_t stream) NOEXCEPT override {
		dnnType *srcData = (dnnType*)reinterpret_cast<const dnnType*>(inputs[0]);
		dnnType *dstData = reinterpret_cast<dnnType*>(outputs[0]);
		checkCuda( cudaMemcpyAsync(dstData, srcData, batchSize*rows*cols*sizeof(dnnType), cudaMemcpyDeviceToDevice, stream));

		checkERROR( cublasSetStream(handle, stream) );	
		for(int i=0; i<batchSize; i++) {
			float const alpha(1.0);
			float const beta(0.0);
			int offset = i*rows*cols;
			checkERROR( cublasSgeam( handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, cols, &alpha, srcData + offset, cols, &beta, srcData + offset, rows, dstData + offset, rows ));
		}
		return 0;
	}
	#else
	virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override {
		dnnType *srcData = (dnnType*)reinterpret_cast<const dnnType*>(inputs[0]);
		dnnType *dstData = reinterpret_cast<dnnType*>(outputs[0]);
		checkCuda( cudaMemcpyAsync(dstData, srcData, batchSize*rows*cols*sizeof(dnnType), cudaMemcpyDeviceToDevice, stream));

		checkERROR( cublasSetStream(handle, stream) );	
		for(int i=0; i<batchSize; i++) {
			float const alpha(1.0);
			float const beta(0.0);
			int offset = i*rows*cols;
			checkERROR( cublasSgeam( handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, cols, &alpha, srcData + offset, cols, &beta, srcData + offset, rows, dstData + offset, rows ));
		}
		return 0;
	}
	#endif 


	virtual size_t getSerializationSize() const NOEXCEPT override {
		return 5*sizeof(int);
	}

	virtual void serialize(void* buffer) const NOEXCEPT override {
		char *buf = reinterpret_cast<char*>(buffer),*a = buf;
		tk::dnn::writeBUF(buf, c);
		tk::dnn::writeBUF(buf, h);
		tk::dnn::writeBUF(buf, w);
		tk::dnn::writeBUF(buf, rows);
		tk::dnn::writeBUF(buf, cols);
		assert(buf == a + getSerializationSize());
	}

    const char* getPluginType() const NOEXCEPT override {
        return "FlattenConcatRT_TKDNN";
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
        assert(nbOutputs == 1 && nbInputs ==1);
        rows = inputDims[0].d[0];
        cols = inputDims[0].d[1] * inputDims[0].d[2];
        c = inputDims[0].d[0] * inputDims[0].d[1] * inputDims[0].d[2];
        h = 1;
        w = 1;
	}

	IPluginV2* clone() const NOEXCEPT override{
	    FlattenConcatRT *p = new FlattenConcatRT();
	    p->setPluginNamespace(mPluginNamespace);
	    return p;
	}

	const char* mPluginNamespace;
	int c, h, w;
	int rows, cols;
	cublasStatus_t stat; 
	cublasHandle_t handle; 
};
