namespace neuronka.dataLoading;
using System;
using System.Collections.Generic;
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
    /// Loads image data from CSV.
    /// Each image is expected to be a flat array of integers (0-255).
    /// Returns a list of image arrays.
    /// </summary>
    public List<int[]> LoadImages(string imagesFilePath)
    {
        var images = new List<int[]>();

        foreach (var line in File.ReadLines(imagesFilePath))
        {
            // Split CSV values and parse them as integers
            int[] pixels = line
                .Split(',', StringSplitOptions.RemoveEmptyEntries)
                .Select(s => int.Parse(s.Trim()))
                .ToArray();

            images.Add(pixels);
        }

        return images;
    }

    /// <summary>
    /// Loads labels from a file (one label per line).
    /// </summary>
    public List<int> LoadLabels(string labelsFilePath)
    {
        var labels = new List<int>();

        foreach (var line in File.ReadLines(labelsFilePath))
        {
            if (int.TryParse(line.Trim(), out int label))
            {
                labels.Add(label);
            }
        }

        return labels;
    }

    /// <summary>
    /// Loads both images and labels together.
    /// Returns a list of tuples: (imageData, label)
    /// </summary>
    public List<(int[] Image, int Label)> LoadDataset(string imagesFilePath,  string labelsFilePath)
    {
        var images = LoadImages(imagesFilePath);
        var labels = LoadLabels(labelsFilePath);

        if (images.Count != labels.Count)
            throw new Exception("Number of images and labels do not match!");

        var data = new List<(int[] Image, int Label)>();

        for (int i = 0; i < images.Count; i++)
        {
            if (images[i].Count() != _expectedImageSize)
            {
                throw new Exception($"Image {i} has invalid size");
            }
            data.Add((images[i], labels[i]));
        }

        return data;
    }
    
    public (List<(int[] Image, int Label)> Train, List<(int[] Image, int Label)> Test) LoadData()
    {
        var trainData = LoadDataset(_trainImagesPath, _trainLabelsPath);
        var testData = LoadDataset(_testImagesPath, _testLabelsPath);
        return (trainData, testData);
    }
}