using System.Collections.Generic;
using System.Linq;
using Drawing;
using Unity.VisualScripting;
using UnityEngine;

namespace Neural_Network{
    public class NetworkConfidenceDisplay : MonoBehaviour{
        public TextAsset networkFile;
        public string predictedLabel;

        public MeshRenderer display;
        public TMPro.TMP_Text labelsUI;
        public TMPro.TMP_Text confidenceUI;

        private ImageLoader _imageLoader;
        private DrawingController _drawingController;
        private NeuralNetwork _network;

        private void Start(){
            _drawingController = FindObjectOfType<DrawingController>();
            _network = NetworkSaveData.LoadNetworkFromData(networkFile.text);
            _imageLoader = FindObjectOfType<ImageLoader>();
        }


        private void Update(){
            RenderTexture digitRenderTexture = _drawingController.RenderOutputTexture();
            Image image = ImageHelper.TextureToImage(digitRenderTexture, 0);
            (int prediction, double[] outputs) = _network.Classify(image.PixelValues);

            UpdateDisplay(image, outputs, prediction);
        }

        private void UpdateDisplay(Image image, IReadOnlyCollection<double> outputs, int prediction){
            predictedLabel = _imageLoader.LabelNames[prediction];
            List<RankedLabel> labels = outputs
                .Select((t, i) => new RankedLabel(){ Name = _imageLoader.LabelNames[i], Score = (float)t }).ToList();
            labels.Sort((a, b) => b.Score.CompareTo(a.Score));
            //Clear the text
            labelsUI.text = "";
            confidenceUI.text = "";
            Color color = Color.white;
            for (int i = 0; i < outputs.Count; i++){
                color.a = Mathf.Clamp(labels[i].Score, 0.25f, 1f);
                string hexCode = color.ToHexString();
                labelsUI.text += $"<color=#{hexCode}>" + labels[i].Name + "\n </color>";
                confidenceUI.text += $"<color=#{hexCode}>" + labels[i].prettyScore + "\n </color>";
            }

            display.material.mainTexture = image.ConvertToTexture2D();
        }

        private struct RankedLabel{
            public string Name;
            public float Score;

            public string prettyScore => $"{Score * 100:0.00}%";
        }
    }
}