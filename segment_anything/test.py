import cv2
import torch
image = cv2.imread("/Users/alexlan/Downloads/a.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
sam = sam_model_registry["default"](checkpoint="/Users/alexlan/Downloads/sam_vit_h_4b8939.pth")
sam.image_encoder = sam.image_encoder.to("mps")

#predictor = SamPredictor(sam, device = 'mps')
mask_generator = SamAutomaticMaskGenerator(sam)
mask_generator.predictor.model = mask_generator.predictor.model.to("mps")
sam.prompt_encoder = sam.prompt_encoder.to("mps")
masks = mask_generator.generate(image)

embed_dim = sam.prompt_encoder.embed_dim
embed_size = sam.prompt_encoder.image_embedding_size
mask_input_size = [4 * x for x in embed_size]


dummy_inputs = {
    "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
    "point_coords": torch.randint(low=0, high=1024, size=(64, 1, 2), dtype=torch.float),
    "point_labels": torch.ones(size=(64,1), dtype=torch.float),
    "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
    "has_mask_input": torch.tensor([0] * 64, dtype=torch.float),
    "orig_im_size": torch.tensor([1838, 1838], dtype=torch.float),
}

from segment_anything.utils.onnx import SamOnnxModel
onnx_model = SamOnnxModel(sam.to("mps"), return_single_mask=False)
onnx_model(**dummy_inputs)
