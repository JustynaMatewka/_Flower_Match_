import onnxruntime as rt
import cv2
import numpy as np

def flower_detector(img_array):
    # Konwersja obrazka czarno-białego na kolorowy
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    
    provider = ['CPUExecutionProvider']
    model = rt.InferenceSession(r"C:\Users\jmatewkx\OneDrive - Intel Corporation\Desktop\Studia\_Flower_match_\emotions_detection\service\core\logic\resnet_18_onnx.onnx", providers=provider)

    test_image = cv2.resize(img_array, (256, 256))
    im = np.float32(test_image)
    img_array = np.expand_dims(im, axis = 0)

    output_names = [output.name for output in model.get_outputs()]
    onnx_pred = model.run(output_names, {'input_1': img_array})

    # CLASS_NAMES = ['bellflower', 'common_daisy', 'rose', 'tulip']
    CLASS_NAMES = ['Dzwonek', 'Stokrotka', 'Róża', 'Tulipan']
    emotion = CLASS_NAMES[np.argmax(onnx_pred[0][0])]

    return {"emotion" : emotion}