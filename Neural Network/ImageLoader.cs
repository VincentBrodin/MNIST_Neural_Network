using System.Collections.Generic;
using UnityEngine;

namespace Neural_Network{
    public class ImageLoader : MonoBehaviour{
        [SerializeField] private int imageSize = 28;
        [SerializeField] private bool greyscale = true;
        [SerializeField] private DataFile[] dataFiles;
        [SerializeField] private string[] labelNames;
        private Image[] _images;

        public int NumberOfImages => _images.Length;
        public int InputSize => imageSize * imageSize * (greyscale ? 1 : 3);
        private int OutputSize => labelNames.Length;
        public string[] LabelNames => labelNames;

        private void Awake(){
            _images = LoadImages();
        }

        public Image GetImage(int i){
            return _images[i];
        }

        public DataPoint[] GetAllData(){
            DataPoint[] allData = new DataPoint[_images.Length];
            for (int i = 0; i < allData.Length; i++){
                allData[i] = DataFromImage(_images[i]);
            }

            return allData;
        }

        private DataPoint DataFromImage(Image image){
            return new DataPoint(image.PixelValues, image.Label, OutputSize);
        }

        private Image[] LoadImages(){
            List<Image> allImages = new();

            foreach (DataFile file in dataFiles){
                IEnumerable<Image> images = LoadImages(file.imageFile.bytes, file.labelFile.bytes);
                allImages.AddRange(images);
            }

            return allImages.ToArray();
        }

        private IEnumerable<Image> LoadImages(IReadOnlyList<byte> imageData, IReadOnlyList<byte> labelData){
            int numberOfChannels = (greyscale) ? 1 : 3;
            int bytesPerImage = imageSize * imageSize * numberOfChannels;

            int numberOfImages = imageData.Count / bytesPerImage;
            int numberOfLabels = labelData.Count / 1;
            
            int dataSetSize = System.Math.Min(numberOfImages, numberOfLabels);
            Image[] images = new Image[dataSetSize];

            const double pixelRangeScale = 1 / 255.0;
            double[] allPixelValues = new double[imageData.Count];

            System.Threading.Tasks.Parallel.For(0, imageData.Count,
                (i) => { allPixelValues[i] = imageData[i] * pixelRangeScale; });

            System.Threading.Tasks.Parallel.For(0, numberOfImages, (imageIndex) => {
                int byteOffset = imageIndex * bytesPerImage;
                double[] pixelValues = new double[bytesPerImage];
                System.Array.Copy(allPixelValues, byteOffset, pixelValues, 0, bytesPerImage);
                Image image = new Image(imageSize, greyscale, pixelValues, labelData[imageIndex]);
                images[imageIndex] = image;
            });

            return images;
        }

        [System.Serializable]
        public struct DataFile{
            public TextAsset imageFile;
            public TextAsset labelFile;
        }
    }
}