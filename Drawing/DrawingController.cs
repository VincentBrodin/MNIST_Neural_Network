using UnityEngine;
using UnityEngine.UI;

namespace Drawing{
    public class DrawingController : MonoBehaviour{
        public int drawResolution = 1024;
        public int outputResolution = 28;

        public BoxCollider2D canvasCollider;
        public ComputeShader drawCompute;
        public float brushRadius;
        public Button clear;
        [Range(0, 1)] public float smoothing;

        private RenderTexture _canvas;
        private RenderTexture _outputCanvas;
        private RenderTexture _croppedCanvas;

        private Camera _cam;
        private Vector2Int _brushCentreOld;
        private ComputeBuffer _boundsBuffer;
        private static readonly int BrushCentre = Shader.PropertyToID("brushCentre");
        private static readonly int BrushCentreOld = Shader.PropertyToID("brushCentreOld");
        private static readonly int BrushRadius = Shader.PropertyToID("brushRadius");
        private static readonly int Smoothing = Shader.PropertyToID("smoothing");
        private static readonly int Resolution1 = Shader.PropertyToID("resolution");
        private static readonly int Mode = Shader.PropertyToID("mode");
        private static readonly int Bounds1 = Shader.PropertyToID("bounds");
        private static readonly int Canvas1 = Shader.PropertyToID("Canvas");

        private void Start(){
            _cam = Camera.main;
            ComputeHelper.CreateRenderTexture(ref _canvas, drawResolution, drawResolution, FilterMode.Bilinear,
                ComputeHelper.defaultGraphicsFormat, "Draw Canvas");
            canvasCollider.gameObject.GetComponent<MeshRenderer>().material.mainTexture = _canvas;

            ComputeHelper.CreateStructuredBuffer<uint>(ref _boundsBuffer, 4);
            _boundsBuffer.SetData(new[]{ drawResolution - 1, 0, drawResolution - 1, 0 });
            drawCompute.SetBuffer(0, Bounds1, _boundsBuffer);
            drawCompute.SetTexture(0, Canvas1, _canvas);
            clear.onClick.AddListener(Clear);
        }


        public RenderTexture RenderOutputTexture(){
            ComputeHelper.CreateRenderTexture(ref _outputCanvas, outputResolution, outputResolution, FilterMode.Point,
                ComputeHelper.defaultGraphicsFormat, "Draw Output");
            RenderTexture source = _canvas;
            RenderTexture downscaleSource = RenderTexture.GetTemporary(source.width, source.height);
            Graphics.Blit(source, downscaleSource);
            int currWidth = source.width / 2;
            while (currWidth > outputResolution * 2){
                RenderTexture temp = RenderTexture.GetTemporary(currWidth, currWidth);
                Graphics.Blit(downscaleSource, temp);
                currWidth /= 2;
                RenderTexture.ReleaseTemporary(downscaleSource);
                downscaleSource = temp;
            }

            Graphics.Blit(downscaleSource, _outputCanvas);
            RenderTexture.ReleaseTemporary(downscaleSource);
            return _outputCanvas;
        }

        private void Update(){
            Vector2 mouseWorld = _cam.ScreenToWorldPoint(Input.mousePosition);

            Bounds canvasBounds = canvasCollider.bounds;

            bool inBounds = canvasBounds.Contains(mouseWorld);
            //if (!inBounds) return;

            float tx = Mathf.InverseLerp(canvasBounds.min.x, canvasBounds.max.x, mouseWorld.x);
            float ty = Mathf.InverseLerp(canvasBounds.min.y, canvasBounds.max.y, mouseWorld.y);

            Vector2Int brushCentre = new((int)(tx * drawResolution), (int)(ty * drawResolution));

            drawCompute.SetInts(BrushCentre, brushCentre.x, brushCentre.y);
            drawCompute.SetInts(BrushCentreOld, _brushCentreOld.x, _brushCentreOld.y);
            drawCompute.SetFloat(BrushRadius, brushRadius);
            drawCompute.SetFloat(Smoothing, smoothing);
            drawCompute.SetInt(Resolution1, drawResolution);
            drawCompute.SetInt(Mode, Input.GetKey(KeyCode.Mouse0) ? 0 : 1);

            if (Input.GetKey(KeyCode.Mouse0)){
                ComputeHelper.Dispatch(drawCompute, drawResolution, drawResolution);
            }

            _brushCentreOld = brushCentre;
        }

        private void Clear(){
            _boundsBuffer.SetData(new[]{ drawResolution - 1, 0, drawResolution - 1, 0 });
            ComputeHelper.CreateRenderTexture(ref _canvas, drawResolution, drawResolution, FilterMode.Bilinear,
                ComputeHelper.defaultGraphicsFormat, "Draw Canvas");
        }

        void OnDestroy(){
            ComputeHelper.Release(_boundsBuffer);
            ComputeHelper.Release(_canvas, _outputCanvas, _croppedCanvas);
        }
    }
}