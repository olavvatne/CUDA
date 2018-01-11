using ManagedCuda;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CUDA_managed_device_info
{
    class Program
    {
        static void Main(string[] args)
        {
            GetInformationAboutDevice();
        }

        // Testing getting device information via managedCuda
        private static void GetInformationAboutDevice()
        {
            // Number of devices
            var deviceCount = CudaContext.GetDeviceCount();
            Console.WriteLine(deviceCount + " Devices");

            if (deviceCount <= 0)
            {
                throw new Exception("No cuda device detected");
            }

            // Pick device based on performance.
            var deviceByFlops = CudaContext.GetMaxGflopsDeviceId();
            Console.WriteLine("Unit {0} has the most Gflops", deviceByFlops);

            var deviceProperties = CudaContext.GetDeviceInfo(deviceByFlops);
            Console.WriteLine("And has the following properties: ");
            Console.WriteLine(deviceProperties.DeviceName);
            Console.WriteLine("Can execute concurrent kernels: " + deviceProperties.ConcurrentKernels);
            Console.WriteLine("Multi processor count: " + deviceProperties.MultiProcessorCount);
            Console.WriteLine("Clockrate (mhz): " + (int)deviceProperties.ClockRate / 1000.0);
            Console.WriteLine("Total global memory (MB): " + deviceProperties.TotalGlobalMemory / 1000000);
            Console.WriteLine("Is integrated: " + deviceProperties.Integrated);
            Console.WriteLine("Max block dimension: " + deviceProperties.MaxGridDim);
            Console.WriteLine("Max block dimension: " + deviceProperties.MaxBlockDim);
            Console.WriteLine("Max threads per block: " + deviceProperties.MaxThreadsPerBlock);
            Console.WriteLine("Max threads per multiprocessor: " + deviceProperties.MaxThreadsPerMultiProcessor);
            Console.WriteLine("Max shared mem block can use (b): " + deviceProperties.SharedMemoryPerBlock);
            Console.WriteLine("If device can do mem copy and kernel execution: " + deviceProperties.GpuOverlap);
            Console.WriteLine("can map memory adress space on host and device: " + deviceProperties.CanMapHostMemory);


        }
    }
}
