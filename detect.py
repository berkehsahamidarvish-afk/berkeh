from ultralytics import YOLO

# بارگذاری مدل پیش‌فرض (بعداً مدل فارسی جایگزین می‌شه)
model = YOLO("yolov8n.pt")

# تست تشخیص روی یه عکس نمونه
results = model("https://ultralytics.com/images/bus.jpg", save=True)

print("تشخیص با موفقیت انجام شد!")
print("نتایج در پوشه runs/detect ذخیره شد")
