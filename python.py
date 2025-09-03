from PIL import Image

# Mở ảnh
img = Image.open(r"C:\Users\Nguyen Thi Thuy\Downloads\z6873248195862_0d0713ccb746af0de425d252dcb87511.jpg")

# Giảm kích thước xuống 50%
w, h = img.size
small_img = img.resize((w // 2, h // 2), Image.LANCZOS)

# Lưu ảnh
small_img.save(r"C:\Users\Nguyen Thi Thuy\Downloads\ảnhmoi.jpg", quality=85)  # quality=85 giúp giảm dung lượng thêm
