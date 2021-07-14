#include <cassert>
#include <iostream>
#include "heap.hpp"
#include "types.hpp"

template<typename T, typename Comp>
bool Heap<T,Comp>::isLeaf(const Int& pos) const
{
    return pos >= (num_>>1) && (pos < num_);
}

template<typename T, typename Comp>
Int Heap<T,Comp>::leftChild(const Int& pos) const
{
    return (pos<<1)+1;
}

template<typename T, typename Comp>
Int Heap<T,Comp>::rightChild(const Int& pos) const
{
    return ((pos+1)<<1);
}

template<typename T, typename Comp>
Int Heap<T,Comp>::parent(const Int& pos) const
{
    return ((pos-1)>>1);
}

template<typename T, typename Comp>
void Heap<T,Comp>::push_back(const T& elem)
{
    Int curr = num_++;
    resize(num_-1, num_);
    data_[curr] = elem;
    while((curr != 0) && (Comp::prior(data_[curr], data_[parent(curr)])))
    {
        swap(data_, curr, parent(curr));
        curr = parent(curr);
    }
}

template<typename T, typename Comp>
T Heap<T,Comp>::pop_back()
{
    if(is_empty())
        assert(1==2);
    swap(data_, 0, --num_);
    if(num_ != 0)
        siftDown(0);
    resize(num_+1, num_);
    return data_[num_];
}

template<typename T, typename Comp>
T Heap<T,Comp>::pop(const Int& i)
{
    Int pos = i;
    if(num_ <= pos)
        assert(1==2);
    if(pos == (num_-1))
        num_--;
    else
    {
        swap(data_, pos, --num_);
        while((pos != 0) && (Comp::prior(data_[pos], data_[parent(pos)])))
        {
            swap(data_, pos, parent(pos));
            pos = parent(pos);
        }
        if(num_ != 0)
            siftDown(pos); 
    }
    resize(num_+1, num_);
    return data_[num_];
}

template<typename T, typename Comp>
bool Heap<T,Comp>::is_empty() const
{
    return num_ == 0;
}

//private function
template<typename T, typename Comp>
void Heap<T,Comp>::siftDown(const Int& i)
{
    Int pos = i;
    while(!isLeaf(pos))
    {
        Int lc = leftChild(pos);
        Int rc = rightChild(pos);
        if((rc < num_) && Comp::prior(data_[rc], data_[lc]))
            lc = rc;
        if((Comp::prior(data_[pos], data_[lc])))
            break;
        swap(data_, pos, lc);
        pos = lc;
    }
}

template<typename T, typename Comp>
void Heap<T,Comp>::heapify()
{
    for(Int i = (num_>>1)-1; i >= 0; --i)
        siftDown(i);
}

//resize the buffer
template<typename T, typename Comp>
void Heap<T,Comp>::resize(const Int& old_size, const Int& new_size)
{
    if((new_size > cap_) || (new_size < (cap_>>2)))
    {
        cap_ = ((new_size > cap_) ? (new_size<<1) : (cap_>>1));
        T* tmp = new T [cap_];
        for(Int i = 0; i < old_size; ++i)
            tmp[i] = data_[i];
    
        delete [] data_;
        data_ = tmp;
    }
}

template<typename T, typename Comp>
void Heap<T,Comp>::push_back(const Heap<T,Comp>& heap)
{
    Int size = heap.size();
    for(Int i = 0; i < size; ++i)
    {
        const T val = heap.at(i);
        push_back(val);
    } 
}

template<typename T, typename Comp>
void Heap<T,Comp>::push_back(T* data, const Int& size)
{
    for(Int i = 0; i < size; ++i)
    {
        T val = data[i];
        push_back(val);
    }
}

template class Heap<Int,Min>;
template class Heap<EdgeTuple,EdgeTupleMin>; 
