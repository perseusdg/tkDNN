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

    const char* getPluginType() const NOEXCEPT override {
        return "ActivationLogisticRT_TKDNN";
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
        size = 1;
        for(int i=0; i<outputDims[0].nbDims; i++)
            size *= outputDims[0].d[i];
    }

    IPluginV2* clone() const NOEXCEPT override{
        ActivationLogisticRT *p = new ActivationLogisticRT;
        p->setPluginNamespace(mPluginNamespace);
        return p;
    }

    const char* mPluginNamespace;
    int size;
};
