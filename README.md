# GPU Computing Gems Jade 2012 Edition - Demo Codes

GPU Computing Gems Jade 2012 Edition实用示例代码

## Content

 1. Large-Scale GPU Search

## Requirements

 1. Cmake 3.8.0+
 2. CUDA 9.0+
 3. GCC 7+

## Compile && Run

```bash
git clone https://github.com/whitelok/gpu-computation-gems-codes.git
cd gpu-computation-gems-codes
mkdir build
cmake ..
make -j8
./bin/[apps you want to run]
```

## 1. Large-Scale GPU Search

大规模GPU查找算法（有序数列查找）

### Performance

 - GPU: V100 * 1
 
|  data type  |  data length  |  keys number  |  time elapsed (ms)  |
|  ----  |  ----  |  ----  |  ----  |
|  uint64_t  |  524288000  |  1024  |  0.020752  |
|  uint64_t  |  524288000  |  16384  |  0.154173  |
|  uint64_t  |  524288000  |  1048576  |  9.50156  |
|  uint64_t  |  1073741824  |  1024  |  0.020496  |
|  uint64_t  |  1073741824  |  16384  |  0.167444  |
|  uint64_t  |  1073741824  |  1048576  |  10.7623  |
|  uint32_t  |  524288000  |  1024  |  0.0195098  |
|  uint32_t  |  524288000  |  16384  |  0.145716  |
|  uint32_t  |  524288000  |  1048576  |  8.38211  |
|  uint32_t  |  1073741824  |  1024  |  0.0195984  |
|  uint32_t  |  1073741824  |  16384  |  0.13724  |
|  uint32_t  |  1073741824  |  1048576  |  9.67058  |