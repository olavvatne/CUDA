using ManagedCuda;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CUDA_managed_csharp
{
    class Program
    {
        static void Main(string[] args)
        {
            // C# Cuda code to call kernel

            int N = 50000;
            int deviceID = 0;
            CudaContext ctx = new CudaContext(deviceID);
            CudaKernel kernel = ctx.LoadKernel("kernel_x64.ptx", "VecAdd");
            int numOfThreads = 256;
            kernel.GridDimensions = (N + numOfThreads - 1) / numOfThreads;
            kernel.BlockDimensions = numOfThreads;

            // allocate memory in host (not gpu)
            var h_A = InitWithData(N, numOfThreads * 4);
            var h_B = InitWithData(N, numOfThreads);

            // Allocate vectors in device memory and copy from host to device.
            CudaDeviceVariable<float> d_A = h_A;
            CudaDeviceVariable<float> d_B = h_B;
            CudaDeviceVariable<float> d_C = new CudaDeviceVariable<float>(N);

            //Invoke kernel
            kernel.Run(d_A.DevicePointer, d_B.DevicePointer, d_C.DevicePointer, N);

            Console.WriteLine("kernel has runeth");
            //Copy from memory of device to host.
            float[] h_C = d_C;
        }


        private static float[] InitWithData(int count, int mod)
        {
            var data = new float[count];
            for (var i = 0; i < count; i++)
            {
                data[i] = i % mod;
            }
            return data;
        }
    }
}
