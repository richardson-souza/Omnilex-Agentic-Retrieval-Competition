import numpy as np

def reproduce():
    print("Testing float32 += float64...")
    a = np.zeros(10, dtype=np.float32)
    b = 1.5 # float64
    try:
        a += b
        print("float32 += float64 worked")
    except TypeError as e:
        print(f"Caught TypeError: {e}")

    print("Testing float32 += float64 (array)...")
    a = np.zeros(10, dtype=np.float32)
    b = np.ones(10, dtype=np.float64) * 1.5
    try:
        a += b
        print("float32 += float64 (array) worked")
    except TypeError as e:
        print(f"Caught TypeError: {e}")

if __name__ == "__main__":
    reproduce()
