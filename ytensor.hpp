#pragma once
#include <vector>
#include <cstddef>
#include <utility>
#include <iostream>
#include <functional>
#include <fstream>

/**
 * @brief 易于使用的张量类。可以处理任意维度的张量。
 * @tparam T 张量元素的数据类型。
 * @tparam dim 张量的维度数。
 */
template <typename T=float, int dim=1>
class YTensor{
public:
    // using iterator = T*;

    // @brief 张量的数据指针。
    T *data;

    // @brief 张量的形状数据。
    int *dimensions;

    // @brief 是否是父张量，只有父张量才能释放内存。
    bool parent;

    ~YTensor();
    YTensor();

    // @brief 构造函数。
    // @param dims 张量的形状。
    // @example YTensor<float, 3> a({3, 4, 5});

    YTensor(std::vector<int> dims);
    // @brief 构造函数。
    // @param args 张量的形状。
    // @example YTensor<float, 3> a(3, 4, 5);
    template <typename... Args>
    YTensor(Args...);

    // @brief 构造函数。
    // @param list 张量的形状。
    // @example YTensor<float, 3> a={3, 4, 5};
    YTensor(std::initializer_list<int> list);

    // @brief 拷贝构造函数。
    // @param other 被拷贝的张量。
    YTensor(const YTensor& other);

    // @brief 获取对应维度的子张量。
    // @param index 张量的索引。
    // @return 对应维度的子张量（非父张量）
    YTensor<T, dim - 1> operator[](int index);

    // @brief 赋值张量，深拷贝，需要张量的形状相同。
    // @param other 被赋值的张量。
    YTensor<T, dim> &operator=(const YTensor& other);

    // @brief 深拷贝。使用等号赋值时完全相等。
    // @return 深拷贝的张量。
    YTensor<T, dim> clone()const;

    // @brief 移动张量的父属性，相当于浅拷贝，移动后原张量在目标张量的作用域外不可以再次使用。(因为没有引用计数所以只能以这个方式实现)
    // @return 移动后的张量。
    YTensor<T, dim> move();

    // @brief 填充张量的所有元素。
    // @param value 填充的值。
    // @return 自身的引用，可以链式操作。
    YTensor<T, dim>& fill(T value);

    // @brief 对张量的所有元素进行变换。
    // @param func 变换函数，输入为张量中的元素值，输出为变换后的元素值。
    // @return 自身的引用，可以链式操作。
    YTensor<T, dim>& transformAll(std::function<T(T&)> func);

    // @brief 张量的点乘，需要张量的形状相同。
    // @return 两个张量的点乘结果。
    YTensor<T, dim> operator*(const YTensor &other)const;

    // @brief 张量的加法，需要张量的形状相同。
    // @return 两个张量的对应位置加法结果。
    YTensor<T, dim> operator+(const YTensor &other)const;

    // @brief 张量的减法，需要张量的形状相同。
    // @return 两个张量的对应位置减法结果。
    YTensor<T, dim> operator-(const YTensor &other)const;

    // @brief 张量的取负。
    // @return 取负后的张量。
    YTensor<T, dim> operator-()const;

    // @brief 张量除法
    // @return 两个张量的对应位置除法结果。
    YTensor<T, dim> operator/(const YTensor &other)const;

    // @brief 获取张量的元素数量
    // @return 张量的元素数量
    inline size_t size() const;

    // @brief 获取张量的形状，推荐使用shape(int)获取。
    // @return 张量的形状，以vector<int>形式存储
    std::vector<int> shape() const;

    // @brief 获取张量的某一维度的大小
    // @param atDim 维度的索引
    inline int shape(int atDim) const;

    // @brief 获取张量的维度数量。
    // @return 张量的维度数量。
    inline size_t shapeSize() const;

    // @brief 获取子张量的元素数量
    // @param atDim 维度的索引
    // @return 此维度下子张量的元素数量
    inline size_t dimSize(int atDim) const;

    // @brief 获取张量的所有维度的元素数量
    // @return 所有维度的元素数量
    inline std::vector<size_t> dimSizes() const;

    // @brief 获取张量在某个位置的元素
    // @param args 张量的坐标
    // @return 张量在此位置的元素
    template<typename... Args> inline T &at(Args... args); 

    // @brief 获取张量在某个位置的元素
    // @param pos 张量的坐标
    // @return 张量在此位置的元素
    inline T &at(std::vector<int> &pos);

    // @brief 获取张量在某个位置的元素
    // @param pos 张量的坐标
    // @return 张量在此位置的元素
    inline T &at(int pos[]);

    // @brief 获取张量在某个位置的元素，等于*(data+dataPos)
    // @param atData 张量的内存索引
    // @return 张量在此位置的元素
    inline T &atData(int dataPos);

    // @brief 将张量的坐标转换为内存索引
    // @param args 张量的坐标
    // @return 张量的内存索引
    template<typename... Args> inline size_t toIndex(Args... args);

    // @brief 将张量的坐标转换为内存索引
    // @param pos 张量的坐标
    // @return 张量的内存索引
    inline size_t toIndex(std::vector<int> &pos);

    // @brief 将张量的坐标转换为内存索引
    // @param pos 张量的坐标
    // @return 张量的内存索引
    inline size_t toIndex(int pos[]);

    // @brief 存储张量到位置
    // @param path 文件路径
    // @return 是否存储成功
    bool save(const std::string path) const;

    // @brief 从位置加载张量
    // @param path 文件路径
    // @return 是否加载成功
    bool load(const std::string path);

    // @brief 输出张量的信息
    template<typename _T,int _D> friend std::ostream &operator<<(std::ostream &os, const YTensor<_T, _D> &tensor);
private:
};

template <typename T>
class YTensor<T, 1>
{
public:
    using iterator = T*;
    T *data;
    int *dimensions;
    bool parent;
    ~YTensor();
    YTensor();
    YTensor(int dim0);
    YTensor(const YTensor &other);
    T &operator[](int index);
    YTensor<T, 1> operator*(const YTensor &other) const;
    YTensor<T, 1> operator+(const YTensor &other)const;
    YTensor<T, 1> operator-(const YTensor &other) const;
    YTensor<T, 1> operator-()const;
    YTensor<T, 1> operator/(const YTensor &other) const;
    YTensor<T, 1> &operator=(const YTensor &other);
    YTensor<T, 1> clone()const;
    YTensor<T, 1> move();
    inline size_t shapeSize() const;
    size_t size() const;
    template <typename _T> friend std::ostream &operator<<(std::ostream &os, const YTensor<_T,1> &tensor);
};


// realize

#define _CRT_SECURE_NO_WARNINGS
#include <algorithm>
#include <map>
#include <deque>
#include <cassert>
#include <iostream>
#include <cstdarg>

template <typename T, int dim>
YTensor<T, dim>::~YTensor(){
    if (parent){
        if (data != nullptr){
            delete[] data;
            data = nullptr;
        }
        if (dimensions != nullptr){
            delete[] dimensions;
            dimensions = nullptr;
        }
    }
}

template <typename T, int dim>
YTensor<T, dim>::YTensor(){
    dimensions = new int[dim];
    std::fill(dimensions, dimensions + dim, 1);
    data = nullptr;
    parent = true;
}

template <typename T, int dim>
YTensor<T, dim>::YTensor(std::vector<int> dims){
    dimensions = new int[dims.size()]; // std::fill
    std::copy(dims.begin(), dims.end(), dimensions);
    parent = true;
    data = new T[size()];
}

template <typename T, int dim>
template <typename... Args>
YTensor<T, dim>::YTensor(Args... args){
    static_assert(sizeof...(args) == dim, "Number of arguments must match the dimension");
    dimensions = new int[dim];
    // auto seq = std::make_index_sequence<sizeof...(args)>();
    int a = 0;
    ((dimensions[a++] = args), ...);
    data = new T[size()];
    parent = true;
}

template <typename T, int dim>
YTensor<T, dim>::YTensor(std::initializer_list<int> list){
    dimensions = new int[list.size()];
    std::copy(list.begin(), list.end(), dimensions);
    data = new T[size()];
    parent = true;
}

template<typename T, int dim>
YTensor<T, dim>::YTensor(const YTensor& other){
    // no need to release memory
    dimensions = new int[dim];
    std::copy(other.dimensions, other.dimensions + dim, dimensions);
    parent = true;
    this->data = new T[size()];
    std::copy(other.data, other.data + other.size(), data);
}

template <typename T, int dim>
YTensor<T, dim - 1> YTensor<T, dim>::operator[](int index){
    index = (index % dimensions[0] + dimensions[0]) % dimensions[0];
    YTensor<T, dim - 1> op;
    delete[] op.dimensions;
    op.dimensions = this->dimensions + 1;
    op.data = this->data + op.size() * index;
    op.parent = false;
    return op;
}

template <typename T, int dim>
YTensor<T, dim> &YTensor<T, dim>::operator=(const YTensor<T, dim> &other){
    if(this == &other){
        return *this;
    }
    if (dim != other.shapeSize()){
        throw std::invalid_argument("YTensor shape size does not match");
    }
    if (parent){
        if (data != nullptr){
            delete[] data;
        }
    }
    std::copy(other.dimensions, other.dimensions + dim, dimensions);
    parent = true;
    this->data = new T[size()];
    std::copy(other.data, other.data + other.size(), data);
    return *this;
}

template<typename T,int dim>
YTensor<T, dim> YTensor<T, dim>::clone()const{
    YTensor<T,dim> op(this->shape());
    std::copy(data,data+size(),op.data);
    op.parent = true;
    return op;
}

template<typename T,int dim>
YTensor<T, dim> YTensor<T, dim>::move(){
    YTensor<T,dim> op;
    op.data = data;
    op.dimensions = dimensions;
    op.parent = true;
    parent = false;
    return op;
}

template<typename T,int dim>
YTensor<T,dim>& YTensor<T,dim>::fill(T value){
    std::fill(data,data+size(),value);
    return *this;
}

template<typename T,int dim>
YTensor<T,dim>& YTensor<T,dim>::transformAll(std::function<T(T&)> func){
    std::transform(data,data+size(),data,func);
    return *this;
}

template <typename T, int dim>
YTensor<T, dim> YTensor<T, dim>::operator*(const YTensor &other)const{
    if (dim != other.shapeSize()){
        throw std::invalid_argument("Dimensions must match");
    }
    YTensor<T, dim> op(this->shape());
    std::transform(this->data, this->data + size(), other.data, op.data, std::multiplies<T>());
    return op;
}

template<typename T, int dim>
YTensor<T, dim> YTensor<T, dim>::operator+(const YTensor &other)const{
    if (dim != other.shapeSize()){
        throw std::invalid_argument("Dimensions must match");
    }
    YTensor<T, dim> op(this->shape());
    std::transform(this->data, this->data + size(), other.data, op.data, std::plus<T>());
    return op;
}

template<typename T, int dim>
YTensor<T, dim> YTensor<T, dim>::operator-(const YTensor &other)const{
    if (dim != other.shapeSize()){
        throw std::invalid_argument("Dimensions must match");
    }
    YTensor<T, dim> op(this->shape());
    std::transform(this->data, this->data + size(), other.data, op.data, std::minus<T>());
    return op;
}

template<typename T, int dim>
YTensor<T, dim> YTensor<T, dim>::operator-()const{
    YTensor<T, dim> op(this->shape());
    std::transform(this->data, this->data + size(), op.data, std::negate<T>());
    return op;
}

template<typename T, int dim>
YTensor<T, dim> YTensor<T, dim>::operator/(const YTensor &other)const{
    if (dim != other.shapeSize()){
        throw std::invalid_argument("Dimensions must match");
    }
    YTensor<T, dim> op(this->shape());
    std::transform(this->data, this->data + size(), other.data, op.data, std::divides<T>());
    return op;
}

template <typename T, int dim>
inline size_t YTensor<T, dim>::size() const{
    size_t op = 1;
    int a = 0;
    for (; a < dim; a++)
    {
        op *= dimensions[a];
    }
    return op;
}

template <typename T, int dim>
std::vector<int> YTensor<T, dim>::shape() const
{
    std::vector<int> op(dim);
    for (int a = 0; a < dim; a++)
    {
        op[a] = dimensions[a];
    }
    return op;
}

template <typename T, int dim>
int YTensor<T, dim>::shape(int atDim) const
{
    atDim = (atDim % dim + dim) % dim;
    return dimensions[atDim];
}

template <typename T, int dim>
template <typename... Args>
T &YTensor<T, dim>::at(Args... args)
{
    if (dim != sizeof...(args))
    {
        throw std::invalid_argument("Number of arguments must match the dimension");
    }
    size_t index = 0;
    int a = 0;
    ((index += args * dimSize(a++)), ...);
    return data[index];
}

template <typename T, int dim>
T &YTensor<T, dim>::at(std::vector<int> &posLoc)
{
    size_t pos = 0;
    int a = 0;
    for (; a < dim; a++)
    {
        pos += posLoc[a] * dimSize(a);
    }
    return *(data + pos);
}

template <typename T, int dim>
T &YTensor<T, dim>::at(int posLoc[])
{
    size_t pos = 0;
    int a = 0;
    for (; a < dim; a++)
    {
        pos += posLoc[a] * dimSize(a);
    }
    return *(data + pos);
}

template<typename T, int dim>
T& YTensor<T,dim>::atData(int atData){
    return *(data+atData);
}

template <typename T, int dim>
size_t YTensor<T, dim>::dimSize(int atDim) const
{
    size_t op = 1;
    for (int a = atDim + 1; a < dim; a++)
    {
        op *= dimensions[a];
    }
    return op;
}

template <typename T, int dim>
std::vector<size_t> YTensor<T, dim>::dimSizes() const
{
    std::vector<size_t> op;
    for (int a = 0; a < dim; a++)
    {
        op.emplace_back(dimSize(a));
    }
    return op;
}

template <typename T, int dim>
size_t YTensor<T, dim>::shapeSize() const
{
    return dim;
}

template <typename T, int dim>
template<typename... Args>
size_t YTensor<T, dim>::toIndex(Args... args){
    if (dim != sizeof...(args))
    {
        throw std::invalid_argument("Number of arguments must match the dimension");
    }
    size_t index = 0;
    std::vector<size_t> sizes=dimSizes();
    int a = 0;
    ((index += args * sizes[a++]), ...);
    return index;
}

template <typename T, int dim>
size_t YTensor<T, dim>::toIndex(std::vector<int> &pos){
    size_t index = 0;
    std::vector<size_t> sizes=dimSizes();
    int a = 0;
    for (; a < dim; a++)    {
        index += pos[a] * sizes[a];
    }
    return index;
}

template <typename T, int dim>
size_t YTensor<T, dim>::toIndex(int pos[]){
    size_t index = 0;
    std::vector<size_t> sizes=dimSizes();
    int a = 0;
    for (; a < dim; a++){
        index += pos[a] * sizes[a];
    }
    return index;
}

template <typename T, int dim>
bool YTensor<T, dim>::save(const std::string path) const{
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()){
        return false;
    }
    file.write(reinterpret_cast<const char *>(dimensions), sizeof(int) * dim);
    file.write(reinterpret_cast<const char *>(data), sizeof(T) * size());
    file.close();
    return true;
}

template <typename T, int dim>
bool YTensor<T, dim>::load(const std::string path){ 
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()){
        return false;
    }
    try {
        file.read(reinterpret_cast<char *>(dimensions), sizeof(int) * dim);
        size_t dataSize = size();
        if (data != nullptr && parent){
            delete[] data;
        }
        data = new T[dataSize];
        file.read(reinterpret_cast<char *>(data), sizeof(T) * dataSize);
        if (file.gcount() != sizeof(T) * dataSize){
            delete[] data;
            data = nullptr;
            file.close();
            return false; // 读取失败
        }
        file.close();
        parent = true;
        return true; // 读取成功
    } catch (const std::exception &e) {
        if (data != nullptr && parent){
            delete[] data;
            data = nullptr;
        }
        file.close();
        return false; // 读取失败
    }
}


template <typename T, int dim>
std::ostream &operator<<(std::ostream &out, const YTensor<T, dim> &tensor)
{
    out << "[YTensor]:<" << typeid(T).name() << ">" << std::endl;
    out << "[itemSize]: " << tensor.size() << std::endl;
    out << "[byteSize]: " << tensor.size() * sizeof(T) << std::endl;
    out << "[shape]: [";
    std::vector<int> dims = tensor.shape();
    for (int a = 0; a < static_cast<int> (dims.size() - 1); a++)
    {
        out << dims[a] << ", ";
    }
    out << dims[static_cast<int> (dims.size()) - 1] << "]" << std::endl;
    out << "[data]:" << std::endl;
    for (int a = 0; a < tensor.size(); a++)
    {
        for (int b = 0; b < static_cast<int>(dims.size()) - 3; b++)
        {
            if (a % tensor.dimSize(b) == 0)
            {
                out << "[";
            }
        }
        for (int b = static_cast<int>(dims.size()) - 3; b < static_cast<int>(dims.size()) - 1; b++)
        {
            if(b<0)continue;
            if (a % tensor.dimSize(b) == 0)
            {
                out << "[";
            }
        }
        out << tensor.data[a] << " ";
        for (int b = 0; b < static_cast<int> (dims.size() - 3); b++)
        {
            if (a % tensor.dimSize(b) == tensor.dimSize(b) - 1)
            {
                out << "]";
            }
        }
        for (int b = static_cast<int> (dims.size()) - 3; b < static_cast<int>(dims.size()) - 1; b++)
        {
            if(b<0)continue;
            if (a % tensor.dimSize(b) == tensor.dimSize(b) - 1)
            {
                out << "]" << std::endl;
            }
        }
    }
    return out;
}

//========================dim==1========================

template <typename T>
T &YTensor<T, 1>::operator[](int index)
{
    return *(data + index);
}

template <typename T>
size_t YTensor<T, 1>::size() const
{
    return dimensions[0];
}

template <typename T>
YTensor<T, 1>::~YTensor()
{
    if (parent)
    {
        if (data != nullptr)
        {
            delete[] data;
            data = nullptr;
        }
        if (dimensions != nullptr){
            delete[] dimensions;
            dimensions = nullptr;
        }
    }
}

template <typename T>
YTensor<T, 1>::YTensor()
{
    dimensions = new int[1];
    dimensions[0] = 1;
    data = nullptr;
    parent = true;
}

template <typename T>
YTensor<T, 1>::YTensor(int dim0)
{
    dimensions = new int[1];
    dimensions[0] = dim0;
    data = new T[dim0];
    parent = true;
}

template<typename T>
YTensor<T, 1>::YTensor(const YTensor &other){
    dimensions = new int[1];
    dimensions[0] = other.dimensions[0];
    parent = true;
    this->data = new T[size()];
    std::copy(other.data, other.data + other.size(), data);
}

template <typename T>
YTensor<T, 1> YTensor<T, 1>::operator*(const YTensor &other)const{
    if(1 != other.shapeSize()){
        throw std::invalid_argument("dim does not match");
    }
    YTensor<T, 1> op(dimensions[0]);
    std::transform(this->data, this->data + dimensions[0], other.data, op.data, std::multiplies<T>());
    return op;
}

template <typename T>
YTensor<T, 1> YTensor<T, 1>::operator+(const YTensor &other)const{
    if(1 != other.shapeSize()){
        throw std::invalid_argument("dim does not match");
    }
    YTensor<T, 1> op(dimensions[0]);
    std::transform(this->data, this->data + dimensions[0], other.data, op.data, std::plus<T>());
    return op;
}

template <typename T>
YTensor<T, 1> YTensor<T, 1>::operator-(const YTensor &other)const{
    if(1 != other.shapeSize()){
        throw std::invalid_argument("dim does not match");
    }
    YTensor<T, 1> op(dimensions[0]);
    std::transform(this->data, this->data + dimensions[0], other.data, op.data, std::minus<T>());
    return op;
}

template <typename T>
YTensor<T, 1> YTensor<T, 1>::operator-()const{
    YTensor<T, 1> op(dimensions[0]);
    std::transform(this->data, this->data + dimensions[0], op.data, std::negate<T>());
    return op;
}

template <typename T>
YTensor<T, 1> YTensor<T, 1>::operator/(const YTensor &other)const{
    if(1 != other.shapeSize()){
        throw std::invalid_argument("dim does not match");
    }
    YTensor<T, 1> op(dimensions[0]);
    std::transform(this->data, this->data + dimensions[0], other.data, op.data, std::divides<T>());
    return op;
}

template<typename T>
YTensor<T, 1> &YTensor<T, 1>::operator=(const YTensor &other){
    if (1 != other.shapeSize()){
        throw std::invalid_argument("YTensor shape size does not match");
    }
    if (parent)
    {
        if (data != nullptr)
        {
            delete[] data;
        }
    }
    std::copy(other.dimensions, other.dimensions + 1, dimensions);
    parent = true;
    this->data = new T[size()];
    std::copy(other.data, other.data + other.size(), data);
    return *this;
}

template<typename T>
YTensor<T, 1> YTensor<T, 1>::clone()const{
    YTensor<T, 1> op(dimensions[0]);
    std::copy(data,data+size(),op.data);
    op.parent = true;
    return op;
}

template<typename T>
YTensor<T, 1> YTensor<T, 1>::move(){
    YTensor<T, 1> op;
    op.data = data;
    op.dimensions = dimensions;
    op.parent = true;
    parent = false;
    return op;
}


template <typename T>
size_t YTensor<T, 1>::shapeSize() const{
    return 1;
}

template <typename T>
std::ostream &operator<<(std::ostream &out, const YTensor<T, 1> &tensor)
{
    out << "[YTensor]:<" << typeid(T).name() << ">" << std::endl;
    out << "[itemSize]: " << tensor.size() << std::endl;
    out << "[byteSize]: " << tensor.size() * sizeof(T) << std::endl;
    out << "[shape]: [";
    std::vector<int> dims = tensor.shape();
    for (int a = 0; a < dims.size() - 1; a++)
    {
        out << a << ", " << dims[a];
    }
    out << dims[dims.size() - 1] << "]" << std::endl;
    out << "[data]:" << std::endl;
    for (int a = 0; a < tensor.size(); a++)
    {
        for (int b = 0; b < dims.size() - 3; b++)
        {
            if (a % dims[b] == 0)
            {
                out << "[";
            }
        }
        for (int b = dims.size() - 3; b < dims.size() - 1; b++)
        {
            if (a % dims[b] == 0)
            {
                out << "[";
            }
        }
        out << tensor.data[a] << " ";
        for (int b = 0; b < dims.size() - 3; b++)
        {
            if (a % dims[b] == dims[b] - 1)
            {
                out << "]";
            }
        }
        for (int b = dims.size() - 3; b < dims.size() - 1; b++)
        {
            if (a % dims[b] == dims[b] - 1)
            {
                out << "]" << std::endl;
            }
        }
    }
    return out;
}

// int main(){
//     YTensor<float, 3> a(8, 8, 8);
//     for(int i=0;i<a.size();i++){
//         *(a.data+i)=i;
//     }
//     a[0][0][0]=10;
//     std::cout<<a.size()<<std::endl;
//     a=a*a;
//     for(int i=0;i<a.shape()[0];i++){
//         for(int j=0;j<a.shape()[1];j++){
//             for(int k=0;k<a.shape()[2];k++){
//                 std::cout<<a[i][j][k]<<" ";
//             }
//             std::cout<<std::endl;
//         }
//         std::cout<<std::endl;
//     }
//     std::cout<<a.at(7,3,4)<<std::endl;
//     std::cout<<a<<std::endl;

//     YTensor<float, 1> b(8);
//     int pin1=b.size();
//     for(int i=0;i<b.size();i++){
//         *(b.data+i)=i;
//     }
//     b[0]=8;
//     for(int i=0;i<b.size();i++){
//         std::cout<<b[i]<<" ";
//     }

//     return 0;
// }
