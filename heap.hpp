#ifndef HEAP_H_
#define HEAP_H_
#include <cstdlib>
#include "types.hpp"
template<typename T, typename Comp>
class Heap
{
  private:
    Int cap_;
    Int num_;
    T* data_;

    void heapify();
    void siftDown(const Int&);
    void resize(const Int&, const Int&);
  
  public:
    Heap() : cap_(1), num_(0), data_(nullptr)
    {
        data_ = new T [cap_];
    };
    Heap(T* data, const Int& num) : cap_(num), num_(num), data_(nullptr) 
    {
        data_ = new T [num];
        for(Int i = 0; i < num; ++i)
            data_[i] = data[i];
        heapify();
    }; 
    ~Heap() { delete [] data_;}

    Int size() const { return num_; }
    T at(const Int& i) const { return data_[i]; }
    T top() const { return data_[0]; }
    
    bool isLeaf(const Int&) const;
    bool is_empty() const;
    Int leftChild(const Int&) const;
    Int rightChild(const Int&) const;
    Int parent(const Int&) const;

    void push_back(const T&);
    T pop_back();
    T pop(const Int&);

    void push_back(const Heap<T,Comp>&);
    void push_back(T*, const Int&);
};

#endif
