from flask import Flask, request, jsonify, render_template
import torch, numpy as np, cv2, os, io, base64
from PIL import Image
import segmentation_models_pytorch as smp

# =========================
# CONFIG DEVICE
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# LOAD MODEL
# =========================
def load_model():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1
    )
    model_path = r"D:\Progamming\Progamming_courses\Python\Segmentation_project\unet_best.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# =========================
# INIT FLASK
# =========================
app = Flask(
    __name__,
    template_folder="../frontend",   # index.html ở đây
    static_folder="../frontend"      # css, js cũng ở đây
)

# =========================
# HELPER FUNCTIONS
# =========================
def preprocess_image(image: Image.Image):
    img = np.array(image)
    img_resized = cv2.resize(img, (256, 256)) / 255.0
    x_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
    return img, x_tensor

def predict_mask(x_tensor, original_img):
    with torch.no_grad():
        pred = model(x_tensor)
        pred = torch.sigmoid(pred).squeeze().cpu().numpy()
    mask = cv2.resize(pred, (original_img.shape[1], original_img.shape[0]))
    mask_bin = (mask > 0.6).astype(np.uint8) * 255

    # Tạo overlay
    mask_color = np.zeros_like(original_img)
    mask_color[mask_bin == 255] = (255, 0, 0)
    overlay = cv2.addWeighted(mask_color, 0.4, original_img, 0.6, 0)
    return mask_bin, overlay

def pil_to_base64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()

# =========================
# ROUTES
# =========================
@app.route("/")
def home():
    test_dir = r"D:\Progamming\Progamming_courses\Python\Segmentation_project\archive\test"
    files = []
    if os.path.exists(test_dir):
        files = [f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    return render_template("index.html", test_files=files)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" in request.files:
        file = request.files["image"]
        image = Image.open(file).convert("RGB")
    elif "test_image" in request.form:
        filename = request.form["test_image"]
        test_dir = r"D:\Progamming\Progamming_courses\Python\Segmentation_project\archive\test"
        path = os.path.join(test_dir, filename)
        image = Image.open(path).convert("RGB")
    else:
        return jsonify({"error": "No image provided"}), 400

    original_img, x_tensor = preprocess_image(image)
    mask_bin, overlay = predict_mask(x_tensor, original_img)

    # Convert sang PIL
    original_pil = Image.fromarray(original_img)
    mask_pil = Image.fromarray(mask_bin)
    overlay_pil = Image.fromarray(overlay)

    return jsonify({
        "real": pil_to_base64(original_pil),   # 🔑 đổi "original" → "real"
        "mask": pil_to_base64(mask_pil),
        "overlay": pil_to_base64(overlay_pil)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
