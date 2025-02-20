using System;
using System.Collections.Generic;
using System.Linq;

public class AnimationSmoother
    {
        public static List<List<float>> SmoothByAveragingPairs(List<List<float>> data)
        {
            if (data.Count < 2) return new List<List<float>>(data);
            var smoothedData = new List<List<float>>(data.Count - 1);
            for (int i = 0; i < data.Count - 1; i++)
            {
                var smoothedFrame = new List<float>(data[0].Count);
                for (int j = 0; j < data[i].Count; j++)
                {
                    smoothedFrame.Add((data[i][j] + data[i + 1][j]) / 2.0f);
                }
                smoothedData.Add(smoothedFrame);
            }
            return smoothedData;
        }

        public static List<List<float>> SmoothByExponentialMovingAverage(List<List<float>> data, float alpha = 0.60f)
        {
            if (data.Count < 1) return new List<List<float>>(data);
            var smoothedData = new List<List<float>>(data.Count);
            for (int i = 0; i < data.Count; i++)
            {
                smoothedData.Add(new List<float>(data[0].Count));
            }
            // Initialize the first frame
            for (int j = 0; j < data[0].Count; j++)
            {
                smoothedData[0].Add(data[0][j]);
            }
            for (int i = 1; i < data.Count; i++)
            {
                var previousFrame = smoothedData[i - 1];
                var currentFrame = smoothedData[i];
                currentFrame.Clear();
                for (int j = 0; j < data[i].Count; j++)
                {
                    float previousValue = previousFrame[j];
                    currentFrame.Add(alpha * data[i][j] + (1 - alpha) * previousValue);
                }
            }
            return smoothedData;
        }

        public static List<List<float>> SmoothByZeroPhaseButterworthFilter(
            List<List<float>> data,
            float cutoffFrequency = 30.0f,
            int samplingRate = 60,
            float smoothingStrength = 0.7f) // NEW: optional parameter to dial back the filter
        {
            if (data.Count < 1)
                return new List<List<float>>(data);

            int paddingSize = 3; // Number of frames to mirror for padding
            var paddedData = AddMirrorPadding(data, paddingSize);

            float rc = 1.0f / (2.0f * (float)Math.PI * cutoffFrequency);
            float dt = 1.0f / samplingRate;
            float alpha = dt / (rc + dt);

            // First pass: forward filtering (sequential due to recursive dependency)
            var forwardFiltered = SmoothByButterworthFilterPass(paddedData, alpha);

            // Reverse the data for the backward pass
            var reversedData = forwardFiltered.AsEnumerable().Reverse().ToList();

            // Second pass: backward filtering (sequential)
            var backwardFiltered = SmoothByButterworthFilterPass(reversedData, alpha);

            // Reverse again to restore original order
            var zeroPhaseData = backwardFiltered.AsEnumerable().Reverse().ToList();

            // Remove the mirror padding
            var filteredData = RemovePadding(zeroPhaseData, paddingSize);

            // NEW: If smoothingStrength is less than 1.0, blend the filtered data with the original data.
            if (smoothingStrength < 1.0f)
            {
                // Preallocate a list with the same number of frames.
                var blendedData = new List<List<float>>(data.Count);
                for (int i = 0; i < data.Count; i++)
                {
                    // Preallocate each frame's list with the proper capacity.
                    blendedData.Add(new List<float>(data[i].Count));
                }

                // Parallelize over frames to blend each frame independently.
                System.Threading.Tasks.Parallel.For(0, data.Count, i =>
                {
                    for (int j = 0; j < data[i].Count; j++)
                    {
                        // Blend: (smoothingStrength * filtered value) + ((1 - smoothingStrength) * original value)
                        blendedData[i].Add(smoothingStrength * filteredData[i][j] + (1.0f - smoothingStrength) * data[i][j]);
                    }
                });
                return blendedData;
            }

            return filteredData;
        }


        // Helper method for single-pass Butterworth filtering
        private static List<List<float>> SmoothByButterworthFilterPass(List<List<float>> data, float alpha)
        {
            var smoothedData = new List<List<float>>(data.Count);
            for (int i = 0; i < data.Count; i++)
            {
                smoothedData.Add(new List<float>(data[0].Count));
            }
            // Initialize first frame
            for (int j = 0; j < data[0].Count; j++)
            {
                smoothedData[0].Add(data[0][j]);
            }
            for (int i = 1; i < data.Count; i++)
            {
                var previousFrame = smoothedData[i - 1];
                var currentFrame = smoothedData[i];
                currentFrame.Clear();
                for (int j = 0; j < data[i].Count; j++)
                {
                    float previousValue = previousFrame[j];
                    currentFrame.Add(previousValue + alpha * (data[i][j] - previousValue));
                }
            }
            return smoothedData;
        }

        // Add mirror padding to the data
        private static List<List<float>> AddMirrorPadding(List<List<float>> data, int paddingSize)
        {
            var paddedData = new List<List<float>>();
            // Mirror the first few frames
            for (int i = paddingSize; i > 0; i--)
            {
                paddedData.Add(new List<float>(data[i - 1]));
            }
            paddedData.AddRange(data);
            // Mirror the last few frames
            for (int i = 0; i < paddingSize; i++)
            {
                paddedData.Add(new List<float>(data[data.Count - 1 - i]));
            }
            return paddedData;
        }

        // Remove the padding from the data
        private static List<List<float>> RemovePadding(List<List<float>> data, int paddingSize)
        {
            return data.Skip(paddingSize).Take(data.Count - 2 * paddingSize).ToList();
        }

        public static List<List<float>> SmoothBySavitzkyGolay(List<List<float>> data, int windowSize = 5, int polynomialOrder = 2)
        {
            if (data.Count < windowSize || windowSize % 2 == 0)
                throw new ArgumentException("Window size must be odd and greater than 1.");

            int halfWindow = windowSize / 2;
            var coefficients = CalculateSavitzkyGolayCoefficients(windowSize, polynomialOrder);
            var smoothedData = new List<List<float>>(data.Count);

            for (int i = 0; i < data.Count; i++)
            {
                var smoothedFrame = new List<float>(data[0].Count);
                for (int j = 0; j < data[i].Count; j++)
                {
                    float sum = 0f;
                    for (int k = -halfWindow; k <= halfWindow; k++)
                    {
                        int idx = Math.Clamp(i + k, 0, data.Count - 1);
                        sum += coefficients[k + halfWindow] * data[idx][j];
                    }
                    smoothedFrame.Add(sum);
                }
                smoothedData.Add(smoothedFrame);
            }
            return smoothedData;
        }

        private static float[] CalculateSavitzkyGolayCoefficients(int windowSize, int polynomialOrder)
        {
            if (windowSize == 5 && polynomialOrder == 2)
                return new float[] { -3f / 35, 12f / 35, 17f / 35, 12f / 35, -3f / 35 };
            if (windowSize == 7 && polynomialOrder == 2)
                return new float[] { -2f / 21, 3f / 21, 6f / 21, 7f / 21, 6f / 21, 3f / 21, -2f / 21 };

            throw new NotImplementedException("Only predefined coefficients for common cases are implemented.");
        }

        public static List<List<float>> SmoothByGaussian(List<List<float>> data, int kernelSize = 5, float sigma = 1.0f)
        {
            if (kernelSize % 2 == 0 || kernelSize < 1)
                throw new ArgumentException("Kernel size must be odd and greater than 0.");

            int halfKernel = kernelSize / 2;
            var kernel = CalculateGaussianKernel(kernelSize, sigma);
            var smoothedData = new List<List<float>>(data.Count);

            for (int i = 0; i < data.Count; i++)
            {
                var smoothedFrame = new List<float>(data[0].Count);
                for (int j = 0; j < data[i].Count; j++)
                {
                    float sum = 0f;
                    for (int k = -halfKernel; k <= halfKernel; k++)
                    {
                        int idx = Math.Clamp(i + k, 0, data.Count - 1);
                        sum += kernel[k + halfKernel] * data[idx][j];
                    }
                    smoothedFrame.Add(sum);
                }
                smoothedData.Add(smoothedFrame);
            }
            return smoothedData;
        }

        private static float[] CalculateGaussianKernel(int size, float sigma)
        {
            float[] kernel = new float[size];
            int halfSize = size / 2;
            float sum = 0f;
            for (int i = -halfSize; i <= halfSize; i++)
            {
                kernel[i + halfSize] = (float)Math.Exp(-0.5 * (i * i) / (sigma * sigma));
                sum += kernel[i + halfSize];
            }
            for (int i = 0; i < size; i++)
            {
                kernel[i] /= sum;
            }
            return kernel;
        }

        public static List<List<float>> SmoothByMedianFilter(List<List<float>> data, int windowSize = 3)
        {
            if (windowSize % 2 == 0 || windowSize < 1)
                throw new ArgumentException("Window size must be odd and greater than 0.");

            int halfWindow = windowSize / 2;
            var smoothedData = new List<List<float>>(data.Count);
            for (int i = 0; i < data.Count; i++)
            {
                var smoothedFrame = new List<float>(data[0].Count);
                for (int j = 0; j < data[i].Count; j++)
                {
                    var window = new List<float>();
                    for (int k = -halfWindow; k <= halfWindow; k++)
                    {
                        int idx = Math.Clamp(i + k, 0, data.Count - 1);
                        window.Add(data[idx][j]);
                    }
                    window.Sort();
                    smoothedFrame.Add(window[window.Count / 2]);
                }
                smoothedData.Add(smoothedFrame);
            }
            return smoothedData;
        }
    }
