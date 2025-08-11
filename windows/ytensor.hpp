#pragma once
#include <vector>
#include <cstddef>
#include <utility>
#include <iostream>
#include <functional>
#include <fstream>

/**
 * @brief ����ʹ�õ������ࡣ���Դ�������ά�ȵ�������
 * @tparam T ����Ԫ�ص��������͡�
 * @tparam dim ������ά������
 */
template <typename T=float, int dim=1>
class YTensor{
public:
    // using iterator = T*;

    // @brief ����������ָ�롣
    T *data;

    // @brief ��������״���ݡ�
    int *dimensions;

    // @brief �Ƿ��Ǹ�������ֻ�и����������ͷ��ڴ档
    bool parent;

    ~YTensor();
    YTensor();

    // @brief ���캯����
    // @param dims ��������״��
    // @example YTensor<float, 3> a({3, 4, 5});

    YTensor(std::vector<int> dims);
    // @brief ���캯����
    // @param args ��������״��
    // @example YTensor<float, 3> a(3, 4, 5);
    template <typename... Args>
    YTensor(Args...);

    // @brief ���캯����
    // @param list ��������״��
    // @example YTensor<float, 3> a={3, 4, 5};
    YTensor(std::initializer_list<int> list);

    // @brief �������캯����
    // @param other ��������������
    YTensor(const YTensor& other);

    // @brief ��ȡ��Ӧά�ȵ���������
    // @param index ������������
    // @return ��Ӧά�ȵ����������Ǹ�������
    YTensor<T, dim - 1> operator[](int index);

    // @brief ��ֵ�������������Ҫ��������״��ͬ��
    // @param other ����ֵ��������
    YTensor<T, dim> &operator=(const YTensor& other);

    // @brief �����ʹ�õȺŸ�ֵʱ��ȫ��ȡ�
    // @return �����������
    YTensor<T, dim> clone()const;

    // @brief �ƶ������ĸ����ԣ��൱��ǳ�������ƶ���ԭ������Ŀ���������������ⲻ�����ٴ�ʹ�á�(��Ϊû�����ü�������ֻ���������ʽʵ��)
    // @return �ƶ����������
    YTensor<T, dim> move();

    // @brief �������������Ԫ�ء�
    // @param value ����ֵ��
    // @return ��������ã�������ʽ������
    YTensor<T, dim>& fill(T value);

    // @brief ������������Ԫ�ؽ��б任��
    // @param func �任����������Ϊ�����е�Ԫ��ֵ�����Ϊ�任���Ԫ��ֵ��
    // @return ��������ã�������ʽ������
    YTensor<T, dim>& transformAll(std::function<T(T&)> func);

    // @brief �����ĵ�ˣ���Ҫ��������״��ͬ��
    // @return ���������ĵ�˽����
    YTensor<T, dim> operator*(const YTensor &other)const;

    // @brief �����ļӷ�����Ҫ��������״��ͬ��
    // @return ���������Ķ�Ӧλ�üӷ������
    YTensor<T, dim> operator+(const YTensor &other)const;

    // @brief �����ļ�������Ҫ��������״��ͬ��
    // @return ���������Ķ�Ӧλ�ü��������
    YTensor<T, dim> operator-(const YTensor &other)const;

    // @brief ������ȡ����
    // @return ȡ�����������
    YTensor<T, dim> operator-()const;

    // @brief ��������
    // @return ���������Ķ�Ӧλ�ó��������
    YTensor<T, dim> operator/(const YTensor &other)const;

    // @brief ��ȡ������Ԫ������
    // @return ������Ԫ������
    inline size_t size() const;

    // @brief ��ȡ��������״���Ƽ�ʹ��shape(int)��ȡ��
    // @return ��������״����vector<int>��ʽ�洢
    std::vector<int> shape() const;

    // @brief ��ȡ������ĳһά�ȵĴ�С
    // @param atDim ά�ȵ�����
    inline int shape(int atDim) const;

    // @brief ��ȡ������ά��������
    // @return ������ά��������
    inline size_t shapeSize() const;

    // @brief ��ȡ��������Ԫ������
    // @param atDim ά�ȵ�����
    // @return ��ά������������Ԫ������
    inline size_t dimSize(int atDim) const;

    // @brief ��ȡ����������ά�ȵ�Ԫ������
    // @return ����ά�ȵ�Ԫ������
    inline std::vector<size_t> dimSizes() const;

    // @brief ��ȡ������ĳ��λ�õ�Ԫ��
    // @param args ����������
    // @return �����ڴ�λ�õ�Ԫ��
    template<typename... Args> inline T &at(Args... args); 

    // @brief ��ȡ������ĳ��λ�õ�Ԫ��
    // @param pos ����������
    // @return �����ڴ�λ�õ�Ԫ��
    inline T &at(std::vector<int> &pos);

    // @brief ��ȡ������ĳ��λ�õ�Ԫ��
    // @param pos ����������
    // @return �����ڴ�λ�õ�Ԫ��
    inline T &at(int pos[]);

    // @brief ��ȡ������ĳ��λ�õ�Ԫ�أ�����*(data+dataPos)
    // @param atData �������ڴ�����
    // @return �����ڴ�λ�õ�Ԫ��
    inline T &atData(int dataPos);

    // @brief ������������ת��Ϊ�ڴ�����
    // @param args ����������
    // @return �������ڴ�����
    template<typename... Args> inline size_t toIndex(Args... args);

    // @brief ������������ת��Ϊ�ڴ�����
    // @param pos ����������
    // @return �������ڴ�����
    inline size_t toIndex(std::vector<int> &pos);

    // @brief ������������ת��Ϊ�ڴ�����
    // @param pos ����������
    // @return �������ڴ�����
    inline size_t toIndex(int pos[]);

    // @brief �洢������λ��
    // @param path �ļ�·��
    // @return �Ƿ�洢�ɹ�
    bool save(const std::string path) const;

    // @brief ��λ�ü�������
    // @param path �ļ�·��
    // @return �Ƿ���سɹ�
    bool load(const std::string path);

    // @brief �����������Ϣ
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
            return false; // ��ȡʧ��
        }
        file.close();
        parent = true;
        return true; // ��ȡ�ɹ�
    } catch (const std::exception &e) {
        if (data != nullptr && parent){
            delete[] data;
            data = nullptr;
        }
        file.close();
        return false; // ��ȡʧ��
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
