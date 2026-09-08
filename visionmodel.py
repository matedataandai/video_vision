import torch
import torchvision.models as models
from torchvision import transforms
class VisionModel():
    def __init__(self):
        self.model = models.resnet50(pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2)
        self.model.load_state_dict(torch.load('model_weights/best_model_noresize.pth', map_location=torch.device("cpu")))
    def predict(self, image):
        self.model.eval()
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()])
        original_size = image.size
        image = transform(image)
        with torch.no_grad():
            image = image.unsqueeze(0) 
            output = self.model(image).cpu().detach().numpy().reshape(-1, 2)
            output[:, 0] *= original_size[0] / 224
            output[:, 1] *= original_size[1] / 224
            return output