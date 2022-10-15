#pragma once

#include "threadfence_wrapper.h"
#include "atomic_lock.h"
#include "gpu.h"
#include "align.h"

#include <string.h>
#include <errno.h>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <cstring>
#include <cassert>
#include <memory>
#include <string>

namespace ipc {

template <bool for_gpu> class ShmChannelReader;
template <bool for_gpu> class ShmChannelWriter;

template <bool for_gpu>
class ShmChannelBase {
  public:
    ShmChannelBase() : shm_(nullptr) {}
    ShmChannelBase(const std::string& name, size_t size = 0);
    ShmChannelBase(ShmChannelBase<for_gpu>* channel) {
        connect(channel);
    }
    ShmChannelBase(size_t size) : ShmChannelBase("", size) {}
    ~ShmChannelBase();

    ShmChannelBase(const ShmChannelBase&) = delete;
    ShmChannelBase<for_gpu>& operator=(const ShmChannelBase<for_gpu>&) = delete;

    ShmChannelBase(ShmChannelBase&&);
    ShmChannelBase<for_gpu>& operator=(ShmChannelBase<for_gpu>&&);

    void connect(std::string name, size_t size = 0);
    void connect(ShmChannelBase<for_gpu>* channel);

    void disconnect();
    bool is_connected();

  protected:
    CUDA_HOSTDEV inline static void* my_memcpy(void* dest, const void* src, size_t count);

    int fd_;
    char* shm_;
    char* ring_buf_;
    size_t size_;
    size_t total_size_;
    bool is_create_;
    std::string name_with_prefix_;

    size_t cached_read_pos_;
    ThreadfenceWrapper<size_t, for_gpu>* read_pos_;
    size_t cached_write_pos_;
    ThreadfenceWrapper<size_t, for_gpu>* write_pos_;

    AtomicLock<for_gpu>* writer_lock_;
};

template <bool for_gpu>
class ShmChannelReader : public ShmChannelBase<for_gpu> {
  public:
    using ShmChannelBase<for_gpu>::ShmChannelBase;

    ShmChannelWriter<for_gpu> fork() {
        ShmChannelWriter<for_gpu> res;
        res.connect(this);
        return res;
    }

    CUDA_HOSTDEV void read(void* buf, size_t size);

    template <typename T>
    CUDA_HOSTDEV void read(T* buf) {
        read(buf, sizeof(T));
    }

    void read(std::string* str) {
        size_t len;
        read(&len);
        str->resize(len);
        read(&((*str)[0]), len);
    }

    template <typename T>
    CUDA_HOSTDEV void read(std::unique_ptr<T>* ptr) {
        T* ptr_tmp;
        read(&ptr_tmp);
        ptr->reset(ptr_tmp);
    }

    CUDA_HOSTDEV bool can_read();
};

template <bool for_gpu>
class ShmChannelWriter : public ShmChannelBase<for_gpu> {
  public:
    using ShmChannelBase<for_gpu>::ShmChannelBase;

    ShmChannelReader<for_gpu> fork() {
        ShmChannelReader<for_gpu> res;
        res.connect(this);
        return res;
    }

    CUDA_HOSTDEV void write(const void* buf, size_t size);

    template <typename T>
    CUDA_HOSTDEV void write(T buf) {
        write(&buf, sizeof(T));
    }

    void write(const std::string& str) {
        write(str.size());
        write(str.c_str(), str.size());
    }

    template <typename T>
    void write(std::unique_ptr<T>&& ptr) {
        T* ptr_tmp = ptr.release();
        write(reinterpret_cast<uintptr_t>(ptr_tmp));
    }

    CUDA_HOSTDEV void acquire_writer_lock();
    CUDA_HOSTDEV void release_writer_lock();
};

using ShmChannelCpuReader = ShmChannelReader<false>;
using ShmChannelCpuWriter = ShmChannelWriter<false>;
using ShmChannelGpuReader = ShmChannelReader<true>;
using ShmChannelGpuWriter = ShmChannelWriter<true>;

template <>
CUDA_HOSTDEV inline void* ShmChannelBase<false>::my_memcpy(void* dest, const void* src, size_t count) {
    return memcpy(dest, src, count);
}

template <>
CUDA_HOSTDEV inline void* ShmChannelBase<true>::my_memcpy(void* dest_, const void* src_, size_t count) {
    volatile char* dest = reinterpret_cast<volatile char*>(dest_);
    volatile const char* src = reinterpret_cast<volatile const char*>(src_);

    for (size_t i = 0; i < count; ++i) {
        dest[i] = src[i];
    }

    return dest_;
}

template <bool for_gpu>
CUDA_HOSTDEV void ShmChannelReader<for_gpu>::read(void* buf, size_t size) {
    size_t size_to_read = size;
    size_t size_read = 0;
    while (size_to_read > 0) {
        while ((this->cached_write_pos_ = this->write_pos_->load()) == this->cached_read_pos_) {}

        size_t final_write_pos;
        if (this->cached_write_pos_ < this->cached_read_pos_) {
            final_write_pos = this->size_;
        } else {
            final_write_pos = this->cached_write_pos_;
        }

        size_t size_can_read = final_write_pos - this->cached_read_pos_;
        size_t size_reading = ((size_to_read < size_can_read) ? size_to_read : size_can_read);

        this->my_memcpy(reinterpret_cast<char*>(buf) + size_read,
                        this->ring_buf_ + this->cached_read_pos_, size_reading);

        size_to_read -= size_reading;
        size_read += size_reading;

        this->cached_read_pos_ += size_reading;
        assert(this->cached_read_pos_ <= this->size_);
        if (this->cached_read_pos_ == this->size_) {
            this->cached_read_pos_ = 0;
        }
        this->read_pos_->store(this->cached_read_pos_);
    }
}

template <bool for_gpu>
CUDA_HOSTDEV void ShmChannelWriter<for_gpu>::write(const void* buf, size_t size) {
    size_t size_to_write = size;
    size_t size_written = 0;

    while (size_to_write > 0) {
        while ((this->cached_write_pos_ + 1) % this->size_ == (this->cached_read_pos_ = this->read_pos_->load())) {}

        size_t size_can_write;
        if (this->cached_read_pos_ <= this->cached_write_pos_) {
            if (this->cached_read_pos_ > 0) {
                size_can_write = this->size_ - this->cached_write_pos_;
            } else {
                size_can_write = this->size_ - this->cached_write_pos_ - 1;
            }
        } else {
            size_can_write = this->cached_read_pos_ - this->cached_write_pos_ - 1;
        }

        size_t size_writing = ((size_to_write < size_can_write) ? size_to_write : size_can_write);

        this->my_memcpy(this->ring_buf_ + this->cached_write_pos_,
                        reinterpret_cast<const char*>(buf) + size_written,
                        size_writing);

        size_to_write -= size_writing;
        size_written += size_writing;

        this->cached_write_pos_ += size_writing;
        assert(this->cached_write_pos_ <= this->size_);
        if (this->cached_write_pos_ == this->size_) {
            this->cached_write_pos_ = 0;
        }
        this->write_pos_->store(this->cached_write_pos_);
    }
}

template <bool for_gpu>
CUDA_HOSTDEV bool ShmChannelReader<for_gpu>::can_read() {
    if (this->cached_write_pos_ != this->cached_read_pos_) {
        return true;
    }
    return this->write_pos_->load() != this->cached_read_pos_;
}

template <bool for_gpu>
CUDA_HOSTDEV void ShmChannelWriter<for_gpu>::acquire_writer_lock() {
    this->writer_lock_->acquire();
}

template <bool for_gpu>
CUDA_HOSTDEV void ShmChannelWriter<for_gpu>::release_writer_lock() {
    this->writer_lock_->release();
}

template <bool for_gpu>
ShmChannelBase<for_gpu>::ShmChannelBase(const std::string& name, size_t size) {
    connect(name, size);
}

template <bool for_gpu>
ShmChannelBase<for_gpu>::~ShmChannelBase() {
    disconnect();
}

template <bool for_gpu>
ShmChannelBase<for_gpu>::ShmChannelBase(ShmChannelBase<for_gpu>&& rhs) {
    *this = std::move(rhs);
}

template <bool for_gpu>
ShmChannelBase<for_gpu>& ShmChannelBase<for_gpu>::operator=(ShmChannelBase<for_gpu>&& rhs) {
    this->fd_ = rhs.fd_;
    this->shm_ = rhs.shm_;
    this->ring_buf_ = rhs.ring_buf_;
    this->size_ = rhs.size_;
    this->total_size_ = rhs.total_size_;
    this->is_create_ = rhs.is_create_;
    this->name_with_prefix_ = rhs.name_with_prefix_;
    this->cached_read_pos_ = rhs.cached_read_pos_;
    this->read_pos_ = rhs.read_pos_;
    this->cached_write_pos_ = rhs.cached_write_pos_;
    this->write_pos_ = rhs.write_pos_;
    this->writer_lock_ = rhs.writer_lock_;

    rhs.shm_ = nullptr;

    return *this;
}

template <bool for_gpu>
void ShmChannelBase<for_gpu>::connect(std::string name, size_t size) {
    shm_ = nullptr;

    if (name != "") {
        is_create_ = (size > 0);
        name_with_prefix_ = "llis:channel:" + name;
        if (is_create_) {
            fd_ = shm_open(name_with_prefix_.c_str(), O_CREAT | O_RDWR, 0600);
        } else {
            fd_ = shm_open(name_with_prefix_.c_str(), O_RDWR, 0600);
        }
        // TODO: error handling

        if (is_create_) {
            size_ = size;
        } else {
            size_t* size_shm_ = reinterpret_cast<size_t*>(mmap(nullptr, sizeof(size_t), PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0));
            size_ = *size_shm_;
            munmap(size_shm_, sizeof(size_t));
        }
    } else {
        is_create_ = true;
        name_with_prefix_ = "";
        size_ = size;
    }

    total_size_ = sizeof(size_t);

    size_t read_pos_pos = next_aligned_pos(total_size_, alignof(ThreadfenceWrapper<size_t, for_gpu>));
    total_size_ = read_pos_pos + sizeof(ThreadfenceWrapper<size_t, for_gpu>);

    size_t write_pos_pos = next_aligned_pos(total_size_, alignof(ThreadfenceWrapper<size_t, for_gpu>));
    total_size_ = write_pos_pos + sizeof(ThreadfenceWrapper<size_t, for_gpu>);

    size_t writer_lock_pos = next_aligned_pos(total_size_, alignof(AtomicLock<for_gpu>));
    total_size_ = writer_lock_pos + sizeof(AtomicLock<for_gpu>);

    size_t ring_buf_offset = total_size_;

    total_size_ += size_;

    if (name_with_prefix_ != "") {
        if (is_create_) {
            if (ftruncate(fd_, total_size_) == -1) {
                fprintf(stdout, "ftruncate error: %s", strerror(errno));
            }
        }
        shm_ = reinterpret_cast<char*>(mmap(nullptr, total_size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0));
    } else {
        shm_ = new char[total_size_];
    }

    /*
    if constexpr (for_gpu) {
        cudaHostRegister(shm_, total_size_, cudaHostRegisterDefault);
    }
    */

    ring_buf_ = shm_ + ring_buf_offset;

    read_pos_ = reinterpret_cast<ThreadfenceWrapper<size_t, for_gpu>*>(shm_ + read_pos_pos);
    write_pos_ = reinterpret_cast<ThreadfenceWrapper<size_t, for_gpu>*>(shm_ + write_pos_pos);
    writer_lock_ = reinterpret_cast<AtomicLock<for_gpu>*>(shm_ + writer_lock_pos);

    if (is_create_) {
        *reinterpret_cast<size_t*>(shm_) = size;
        read_pos_->store(0);
        cached_read_pos_ = 0;
        write_pos_->store(0);
        cached_write_pos_ = 0;
        writer_lock_->init();
    } else {
        cached_read_pos_ = read_pos_->load();
        cached_write_pos_ = write_pos_->load();
    }
}

template <bool for_gpu>
void ShmChannelBase<for_gpu>::connect(ShmChannelBase<for_gpu>* channel) {
    fd_ = -1;
    shm_ = channel->shm_;
    ring_buf_ = channel->ring_buf_;
    size_ = channel->size_;
    total_size_ = channel->total_size_;
    is_create_ = false;
    name_with_prefix_ = channel->name_with_prefix_;
    cached_read_pos_ = channel->cached_read_pos_;
    read_pos_ = channel->read_pos_;
    cached_write_pos_ = channel->cached_write_pos_;
    write_pos_ = channel->write_pos_;
    writer_lock_ = channel->writer_lock_;
}

template <bool for_gpu>
void ShmChannelBase<for_gpu>::disconnect() {
    if (is_connected()) {
        if (name_with_prefix_ != "") {
            munmap(shm_, total_size_);
            if (fd_ != -1) {
                close(fd_);
            }
            if (is_create_) {
                shm_unlink(name_with_prefix_.c_str());
            }
        } else {
            if (is_create_) {
                delete[] shm_;
            }
        }
        shm_ = nullptr;
    }
}

template <bool for_gpu>
bool ShmChannelBase<for_gpu>::is_connected() {
    return shm_ != nullptr;
}

template class ShmChannelBase<true>;
#ifndef __CUDA_ARCH__
template class ShmChannelBase<false>;
#endif

}

