# Simple Chatbot with TensorFlow

This Python project implements a simple chatbot using TensorFlow. The chatbot is trained on a dataset provided in the `intents.json` file to understand and respond to user queries based on predefined patterns.

## Requirements
- Python 3.x
- TensorFlow
- NLTK
- NumPy

## Usage
1. Clone or download the repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the `chatbot.py` script.

## Training Data Format
The training data should be provided in JSON format (`intents.json`). Each intent should contain patterns and corresponding responses. Example:

```json
{
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["Hi", "Hello", "Hey"],
            "responses": ["Hello!", "Hi there!", "Hey!"]
        },
        ...
    ]
}
```

## Model Architecture
The chatbot model architecture consists of a neural network with an input layer, a hidden layer with ReLU activation, and an output layer with softmax activation.

## Training
The training process involves tokenizing the patterns, preparing the training data, defining the model, compiling it, and then fitting the model to the training data. Alternatively, you can load a pre-trained model from the `chatbot_model.h5` file.

## Response Generation
The chatbot generates responses by tokenizing the user input, converting it into a bag of words, and then passing it through the trained model to predict the appropriate response.

## Example
Here's an example of how to interact with the chatbot:

```python
python chatbot.py
```

## Additional Notes
- Make sure to provide a well-structured `intents.json` file for effective training.
- The provided `chatbot_model.h5` file contains a pre-trained model. Set `load_saved_model` to `False` in the script to train a new model.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
