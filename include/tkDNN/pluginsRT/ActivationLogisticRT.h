#include<cassert>
#include "../kernels.h"

class ActivationLogisticRT : public IPluginV2 {

public:
    ActivationLogisticRT() {


    }

    ~ActivationLogisticRT(){

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

    #if NV_TENSORRT_MAJOR >=8
    virtual int32_t enqueue(int32_t batchSize,void const*const* inputs,void *const *outputs,void* workspace,cudaStream_t stream) NOEXCEPT override{
              activationLOGISTICForward((dnnType*)reinterpret_cast<const dnnType*>(inputs[0]),
                                  reinterpret_cast<dnnType*>(outputs[0]), batchSize*size, stream);
        return 0;
    }
    #else
    virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override {

        activationLOGISTICForward((dnnType*)reinterpret_cast<const dnnType*>(inputs[0]),
                                  reinterpret_cast<dnnType*>(outputs[0]), batchSize*size, stream);
        return 0;
    }
    #endif 

    virtual size_t getSerializationSize() const NOEXCEPT override {
        return 1*sizeof(int);
    }

    virtual void serialize(void* buffer) const NOEXCEPT override {
        char *buf = reinterpret_cast<char*>(buffer);
        tk::dnn::writeBUF(buf, size);
    }

    int size;
};
