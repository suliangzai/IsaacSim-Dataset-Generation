import numpy as np

def read_camera_matrices_dic(matrices_path):
    data = np.load(matrices_path, allow_pickle=True)
    intrinsic_matrix = data['intrinsic']
    extrinsic_matrix = data['extrinsic']
    return {"intrinsic": intrinsic_matrix, "extrinsic": extrinsic_matrix}

    # print("Intrinsic Matrix:", intrinsic_matrix)
    # print("Extrinsic Matrix:", extrinsic_matrix)

if __name__ == "__main__":
    matrices_path = "/home/ja/Projects/Simulation/data/camera/0/camera_0005.npz"
    matrices_dic = read_camera_matrices_dic(matrices_path)
    print(matrices_dic["intrinsic"])
    print(matrices_dic["extrinsic"])
