using UnityEngine;

namespace Neural_Network{
    public class Image{
        public readonly int Size;
        public readonly int NumberOfPixels;
        public readonly bool Greyscale;

        public readonly double[] PixelValues;
        public readonly int Label;

        public Image(int size, bool greyscale, double[] pixelValues, int label){
            Label = label;

            Size = size;
            NumberOfPixels = size * size;
            PixelValues = pixelValues;

            Greyscale = greyscale;
        }

        public Image(int size, bool greyscale, int label){
            Label = label;

            Size = size;
            NumberOfPixels = size * size;
            PixelValues = new double[NumberOfPixels];

            Greyscale = greyscale;
        }

        public int GetFlatIndex(int x, int y){
            return y * Size + x;
        }

        public double Sample(double u, double v){
            u = System.Math.Max(System.Math.Min(1, u), 0);
            v = System.Math.Max(System.Math.Min(1, v), 0);

            double texX = u * (Size - 1);
            double texY = v * (Size - 1);

            int indexLeft = (int)(texX);
            int indexBottom = (int)(texY);
            int indexRight = System.Math.Min(indexLeft + 1, Size - 1);
            int indexTop = System.Math.Min(indexBottom + 1, Size - 1);

            double blendX = texX - indexLeft;
            double blendY = texY - indexBottom;

            double bottomLeft = PixelValues[GetFlatIndex(indexLeft, indexBottom)];
            double bottomRight = PixelValues[GetFlatIndex(indexRight, indexBottom)];
            double topLeft = PixelValues[GetFlatIndex(indexLeft, indexTop)];
            double topRight = PixelValues[GetFlatIndex(indexRight, indexTop)];

            double valueBottom = bottomLeft + (bottomRight - bottomLeft) * blendX;
            double valueTop = topLeft + (topRight - topLeft) * blendX;
            double interpolatedValue = valueBottom + (valueTop - valueBottom) * blendY;
            return interpolatedValue;
        }

        public Texture2D ConvertToTexture2D(){
            Texture2D texture = new Texture2D(Size, Size);
            ConvertToTexture2D(ref texture);
            return texture;
        }

        public void ConvertToTexture2D(ref Texture2D texture){
            if (texture == null || texture.width != Size || texture.height != Size){
                texture = new Texture2D(Size, Size);
            }

            texture.filterMode = FilterMode.Point;

            Color[] colors = new Color[NumberOfPixels];
            for (int i = 0; i < NumberOfPixels; i++){
                if (Greyscale){
                    float v = (float)PixelValues[i];
                    colors[i] = new Color(v, v, v);
                }
                else{
                    float r = (float)PixelValues[i * 3 + 0];
                    float g = (float)PixelValues[i * 3 + 1];
                    float b = (float)PixelValues[i * 3 + 2];
                    colors[i] = new Color(r, g, b);
                }
            }

            texture.SetPixels(colors);
            texture.Apply();
        }
    }
}