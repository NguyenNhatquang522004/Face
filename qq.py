import pickle

OUTPUT_MODEL = r'C:\Dataset\FaceRecognition\classifier\trained_classifier.pkl'

# Nạp lại mô hình từ file
with open(OUTPUT_MODEL, 'rb') as file:
    model = pickle.load(file)

# In ra kiểu dữ liệu của mô hình
print(f"Model type: {type(model)}")

# Kiểm tra các thuộc tính của mô hình
if hasattr(model, "get_params"):
    print("Model parameters:", model.get_params())
if hasattr(model, "classes_"):
    print("Classes:", model.classes_)
    
if hasattr(model, "support_vectors_"):
    print("Support Vectors (SVM):", model.support_vectors_)
