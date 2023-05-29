import torch
from PIL import Image
import numpy as np
import supervision as sv
from .helpers import get_color, find_closest_rgb_name


# Classe responsavel para processar o frame recebido do frontend
class ProcessFile:
    def __init__(self, model, processor, device, colors_dict):
        self.model = model
        self.processor = processor
        self.device = device
        self.colors_dict = colors_dict

    def run(self, file):
        image = Image.open(file)
        image = np.array(image)

        try:
            bb_annotations = sv.BoxAnnotator()

            with torch.no_grad():

                # processa a imagem
                inputs = self.processor(
                    images=image, return_tensors='pt').to(self.device)

                # passa na network
                model_outputs = self.model(**inputs)
                # pos processamento
                target_sizes = torch.tensor([image.shape[:2]]).to(self.device)

                # converte o output retornando os bounding_boxes
                results = self.processor.post_process_object_detection(
                    outputs=model_outputs,
                    threshold=0.5,
                    target_sizes=target_sizes
                )[0]

            # faz as anotacoes
            detections = sv.Detections.from_transformers(
                transformers_results=results).with_nms(threshold=0.5)
            if (len(detections) == 0):
                print("Nenhuma deteccao encontrada")
                return image

            labels = []
            response_detec = []
            for idx, elem in enumerate(detections):
                xyxy, confidence, _, _ = elem
                bounding_box_coords = xyxy
                x1, y1, x2, y2 = int(bounding_box_coords[0]), int(bounding_box_coords[1]), int(
                    bounding_box_coords[2]), int(bounding_box_coords[3])
                cropped_image = image.copy()[y1:y2, x1:x2]
                dominant_color = get_color(cropped_image)
                print(f"imagem:{idx} cor_dominante: {dominant_color}")

                color = find_closest_rgb_name([
                    dominant_color[2], dominant_color[1], dominant_color[0]], self.colors_dict)
                print(f"image:{idx} cor_identificada: {color}")
                labels.append(f"{color.capitalize()} {confidence:.2f}")
                response_detec.append({
                    'detection_number': idx + 1,
                    'color': color.capitalize(),
                    'color_rgb': {
                        'r': str(dominant_color[2]),
                        'g': str(dominant_color[1]),
                        'b': str(dominant_color[0])
                    },
                    'confidence': str(confidence)
                })

            return bb_annotations.annotate(scene=image.copy(), detections=detections, labels=labels), response_detec
        except Exception as e:
            print("Erro desconhecido", str(e))
            return image, [{
                'detection_number': None,
                'color': None,
                'color_rgb': {
                    'r': None,
                    'g': None,
                    'b': None
                },
                'confidence': None
            }]
