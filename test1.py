import pickle

with open("models/face_encodings.pkl", "rb") as f:
    encodings, names = pickle.load(f)

print(f"Total encodings: {len(encodings)}")
print(f"Names: {names}")
print(f"Shape of first encoding: {len(encodings[0])}")