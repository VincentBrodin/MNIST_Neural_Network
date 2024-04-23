namespace Neural_Network{
	[System.Serializable]
	public class NetworkSaveData
	{

		public int[] layerSizes;
		public ConnectionSaveData[] connections;
		public Cost.CostType costFunctionType;

		// Load network from saved data
		public NeuralNetwork LoadNetwork()
		{
			NeuralNetwork network = new NeuralNetwork(layerSizes);
			for (int i = 0; i < network.Layers.Length; i++)
			{
				ConnectionSaveData loadedConnection = connections[i];

				System.Array.Copy(loadedConnection.weights, network.Layers[i].Weights, loadedConnection.weights.Length);
				System.Array.Copy(loadedConnection.biases, network.Layers[i].Biases, loadedConnection.biases.Length);
				network.Layers[i].Activation = Activation.GetActivationFromType(loadedConnection.activationType);
			}
			network.SetCostFunction(Cost.GetCostFromType((Cost.CostType)costFunctionType));

			return network;
		}

		// Load save data from file
		public static NeuralNetwork LoadNetworkFromFile(string path)
		{
			using (var reader = new System.IO.StreamReader(path))
			{
				string data = reader.ReadToEnd();
				return LoadNetworkFromData(data);
			}
		}

		public static NeuralNetwork LoadNetworkFromData(string loadedData)
		{
			return UnityEngine.JsonUtility.FromJson<NetworkSaveData>(loadedData).LoadNetwork();
		}

		public static string SerializeNetwork(NeuralNetwork network)
		{
			NetworkSaveData saveData = new NetworkSaveData();
			saveData.layerSizes = network.LayerSizes;
			saveData.connections = new ConnectionSaveData[network.Layers.Length];
			saveData.costFunctionType = (Cost.CostType)network.Cost.CostFunctionType();

			for (int i = 0; i < network.Layers.Length; i++)
			{
				saveData.connections[i].weights = network.Layers[i].Weights;
				saveData.connections[i].biases = network.Layers[i].Biases;
				saveData.connections[i].activationType = network.Layers[i].Activation.GetActivationType();
			}
			return UnityEngine.JsonUtility.ToJson(saveData);
		}

		public static void SaveToFile(string networkSaveString, string path)
		{
			using (var writer = new System.IO.StreamWriter(path))
			{
				writer.Write(networkSaveString);
			}
		}


		public static void SaveToFile(NeuralNetwork network, string path)
		{
			using (var writer = new System.IO.StreamWriter(path))
			{
				writer.Write(SerializeNetwork(network));
			}
		}


		[System.Serializable]
		public struct ConnectionSaveData
		{
			public double[] weights;
			public double[] biases;
			public Activation.ActivationType activationType;
		}
	}
}
