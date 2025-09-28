namespace neuronka.dataLoading;
using System;
using System.IO;
using System.Linq;

public class DataLoader
{
    private readonly string _trainImagesPath;
    private readonly string _trainLabelsPath;
    private readonly string _testImagesPath;
    private readonly string _testLabelsPath;
    private readonly int _expectedImageSize = 784;

    public DataLoader(string projectRoot)
    {
        try
        {
            _trainImagesPath = Path.Combine(projectRoot, "data", "fashion_mnist_train_vectors.csv");
            _trainLabelsPath = Path.Combine(projectRoot, "data", "fashion_mnist_train_labels.csv");
            _testImagesPath = Path.Combine(projectRoot, "data", "fashion_mnist_test_vectors.csv");
            _testLabelsPath = Path.Combine(projectRoot, "data", "fashion_mnist_test_labels.csv");
        }
        catch(Exception e)
        {
            throw new Exception("Error while reading data file", e);
        }

    }

    /// <summary>
    /// Loads images from CSV into a float[,] array of shape (numFeatures, numSamples)
    /// </summary>
    public float[,] LoadImages(string imagesFilePath)
    {
        var lines = File.ReadLines(imagesFilePath).ToArray();
        int numSamples = lines.Length;
        var images = new float[_expectedImageSize, numSamples];

        for (int j = 0; j < numSamples; j++)
        {
            var pixels = lines[j]
                .Split(',', StringSplitOptions.RemoveEmptyEntries)
                .Select(s => float.Parse(s.Trim()))
                .ToArray();

            if (pixels.Length != _expectedImageSize)
                throw new Exception($"Image {j} has invalid size");

            for (int i = 0; i < _expectedImageSize; i++)
            {
                images[i, j] = pixels[i] / 255f;
            }
        }

        return images;
    }

    /// <summary>
    /// Loads labels into an int[] array
    /// </summary>
    public int[] LoadLabels(string labelsFilePath)
    {
        var lines = File.ReadLines(labelsFilePath).ToArray();
        var labels = new int[lines.Length];

        for (int i = 0; i < lines.Length; i++)
        {
            if (!int.TryParse(lines[i].Trim(), out labels[i]))
                throw new Exception($"Invalid label at line {i}");
        }

        return labels;
    }

    /// <summary>
    /// Loads both images and labels together
    /// Returns (images, labels)
    /// </summary>
    public (float[,] Images, int[] Labels) LoadDataset(string imagesFilePath, string labelsFilePath)
    {
        var images = LoadImages(imagesFilePath);
        var labels = LoadLabels(labelsFilePath);

        if (images.GetLength(1) != labels.Length)
            throw new Exception("Number of images and labels do not match!");

        return (images, labels);
    }

    /// <summary>
    /// Loads train and test datasets
    /// </summary>
    public ((float[,] Images, int[] Labels) Train, (float[,] Images, int[] Labels) Test) LoadData()
    {
        var trainData = LoadDataset(_trainImagesPath, _trainLabelsPath);
        var testData = LoadDataset(_testImagesPath, _testLabelsPath);
        return (trainData, testData);
    }
}
