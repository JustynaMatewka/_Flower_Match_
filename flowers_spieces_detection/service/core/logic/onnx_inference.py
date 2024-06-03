import onnxruntime as rt
import cv2, io, base64, time
import numpy as np
import matplotlib.pyplot as plt

def flower_detector(img_array):
    # Konwersja obrazka czarno-białego na kolorowy
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    
    # Podłączenie modelu
    provider = ['CPUExecutionProvider']
    model = rt.InferenceSession(r"C:\Users\hp\Studia\Projekt\model\resnet18\resnet_18_onnx.onnx", providers=provider)

    # Obróbka zdjęcia
    test_image = cv2.resize(img_array, (256, 256))
    im = np.float32(test_image)
    img_array = np.expand_dims(im, axis = 0)

    # Uruchomienie modelu i pomiar czasu dla tego polecenia
    output_names = [output.name for output in model.get_outputs()]
    start_time = time.time()
    onnx_pred = model.run(output_names, {'input_1': img_array})
    end_time = time.time()

    execution_time = (end_time - start_time) * 60

    # Przypisanie wartości do zwrócenia na podstawie predykcji
    CLASS_NAMES = ['Bellflower', 'Common_daisy', 'Rose', 'Tulip']
    # CLASS_NAMES = ['Dzwonek', 'Stokrotka', 'Róża', 'Tulipan']
    if max(onnx_pred[0][0]) < 0.85:
        emotion = "Not recognized"
    else:
        emotion = CLASS_NAMES[np.argmax(onnx_pred[0][0])]
    
    # Generowanie wykresu słupkowego
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(CLASS_NAMES, onnx_pred[0][0], color='darkgreen', edgecolor='darkgreen')
    ax.set_xlabel('Class')
    ax.set_ylabel('Probability')

    # Konwersja wykresu na obraz base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.read()).decode('utf-8')

    return {"emotion": emotion, "plot": plot_base64, "time": execution_time}