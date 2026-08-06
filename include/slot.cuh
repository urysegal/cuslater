#include "cudapp.cuh"
#include "number.cuh"

using namespace cuslater;
struct Slot {
    cudapp::CudaStream stream;
    CudaNumber number;      // I can choose which stream to use for async operations on this number
    cudapp::CudaEvent done; // explain why this is needed
    bool              pending = false;
    real_t            result;
    real_t            weight;

    Slot() :
        stream(cudapp::CudaStream::Builder().withNonBlocking(true).build()),
        done(cudapp::CudaEvent::Builder().withDisableTiming(true).build()) {}

    void setZero() {
        number.setZero(stream);
    }

    void markDone() {
        stream.recordEvent(done);
        pending = true;
    }

    void wait() {
        done.join();
        number.get(result, stream);
    }
};

struct SlotPair {
    Slot slots[2];
    int  current = 0;

    Slot& getCurrent() {
        return slots[current];
    }

    Slot& getNext() {
        return slots[(current + 1) % 2];
    }

    void advance() {
        current = (current + 1) % 2;
    }

    template<class F1, class F2>
    void execute(F1 task, F2 consumer) {
        Slot& current = getCurrent();
        Slot& prev    = getNext();

        if (prev.pending) {
            prev.wait();
            consumer(prev);
            prev.pending = false;
        }

        current.setZero();
        task(current);
        current.markDone();
        advance();
    }

    template<class Func>
    void drain(Func consumer) {
        for (int i = 0; i < 2; ++i) {
            Slot& slot = slots[i];
            if (slot.pending) {
                slot.wait();
                consumer(slot);
                slot.pending = false;
            }
        }
    }
};